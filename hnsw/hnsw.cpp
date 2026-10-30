#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace std;

// Cosine distance between two vectors
float cosine_distance(const vector<float>& a, const vector<float>& b) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < (int)a.size(); i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na == 0 || nb == 0) return 1.0f;
    return 1.0f - dot / (sqrt(na) * sqrt(nb));
}

struct Node {
    vector<float> vec;          // the embedding vector
    string text;                // the original chunk text
    vector<vector<int>> neighbors; // neighbors at each layer
};

class HNSW {
public:
    int M;          // max neighbors per node per layer
    int ef;         // search beam width
    int maxLayer;
    int entryPoint;
    mt19937 rng;
    vector<Node> nodes;

    HNSW(int M = 16, int ef = 50)
        : M(M), ef(ef), maxLayer(0), entryPoint(-1), rng(42) {}

    // Randomly assign a layer to a new node (higher layers are rarer)
    int randomLayer() {
        uniform_real_distribution<float> dist(0.0f, 1.0f);
        int layer = 0;
        while (dist(rng) < 0.5f && layer < 16) layer++;
        return layer;
    }

    // Greedy search from a start node down to target layer
    vector<pair<float,int>> searchLayer(
        const vector<float>& query, int start, int layer, int ef_search
    ) {
        vector<bool> visited(nodes.size(), false);
        priority_queue<pair<float,int>> worst; // max-heap (farthest first)
        priority_queue<pair<float,int>, vector<pair<float,int>>, greater<>> best; // min-heap

        float d = cosine_distance(query, nodes[start].vec);
        worst.push({d, start});
        best.push({d, start});
        visited[start] = true;

        while (!best.empty()) {
            auto [cd, curr] = best.top(); best.pop();
            if (cd > worst.top().first) break; // can't improve

            if (layer < (int)nodes[curr].neighbors.size()) {
                for (int nb : nodes[curr].neighbors[layer]) {
                    if (!visited[nb]) {
                        visited[nb] = true;
                        float nd = cosine_distance(query, nodes[nb].vec);
                        if (worst.size() < (size_t)ef_search || nd < worst.top().first) {
                            best.push({nd, nb});
                            worst.push({nd, nb});
                            if ((int)worst.size() > ef_search) worst.pop();
                        }
                    }
                }
            }
        }

        vector<pair<float,int>> results;
        while (!worst.empty()) { results.push_back(worst.top()); worst.pop(); }
        sort(results.begin(), results.end());
        return results;
    }

    void insert(const vector<float>& vec, const string& text) {
        int id = nodes.size();
        int layer = randomLayer();

        Node node;
        node.vec = vec;
        node.text = text;
        node.neighbors.resize(layer + 1);
        nodes.push_back(node);

        if (entryPoint == -1) {
            entryPoint = id;
            maxLayer = layer;
            return;
        }

        int curr = entryPoint;

        // Traverse from top layer down to layer+1 (greedy, ef=1)
        for (int l = maxLayer; l > layer; l--) {
            auto res = searchLayer(vec, curr, l, 1);
            curr = res[0].second;
        }

        // From layer down to 0, connect neighbors
        for (int l = min(layer, maxLayer); l >= 0; l--) {
            auto candidates = searchLayer(vec, curr, l, ef);
            int take = min(M, (int)candidates.size());
            for (int i = 0; i < take; i++) {
                int nb = candidates[i].second;
                nodes[id].neighbors[l].push_back(nb);
                nodes[nb].neighbors[l].push_back(id); // bidirectional
                // Trim neighbors if over M, keeping the M closest to nb
                // (cutting the tail would always drop the link to id just added)
                auto& nbList = nodes[nb].neighbors[l];
                if ((int)nbList.size() > M) {
                    const vector<float>& base = nodes[nb].vec;
                    vector<pair<float,int>> scored;
                    for (int x : nbList) scored.push_back({cosine_distance(base, nodes[x].vec), x});
                    sort(scored.begin(), scored.end());
                    nbList.clear();
                    for (int i = 0; i < M; i++) nbList.push_back(scored[i].second);
                }
            }
            if (!candidates.empty()) curr = candidates[0].second;
        }

        if (layer > maxLayer) { maxLayer = layer; entryPoint = id; }
    }

    // Return top-k nearest chunks to query
    vector<pair<float,string>> search(const vector<float>& query, int k = 5) {
        if (entryPoint == -1) return {};
        int curr = entryPoint;
        for (int l = maxLayer; l > 0; l--) {
            auto res = searchLayer(query, curr, l, 1);
            curr = res[0].second;
        }
        auto res = searchLayer(query, curr, 0, ef);
        vector<pair<float,string>> out;
        for (int i = 0; i < min(k, (int)res.size()); i++)
            out.push_back({res[i].first, nodes[res[i].second].text});
        return out;
    }

    // Save index to file
    void save(const string& path) {
        ofstream f(path, ios::binary);
        int n = nodes.size();
        f.write((char*)&n, sizeof(int));
        f.write((char*)&maxLayer, sizeof(int));
        f.write((char*)&entryPoint, sizeof(int));
        for (auto& node : nodes) {
            int vs = node.vec.size();
            f.write((char*)&vs, sizeof(int));
            f.write((char*)node.vec.data(), vs * sizeof(float));
            int ts = node.text.size();
            f.write((char*)&ts, sizeof(int));
            f.write(node.text.c_str(), ts);
            int ls = node.neighbors.size();
            f.write((char*)&ls, sizeof(int));
            for (auto& layer : node.neighbors) {
                int ns = layer.size();
                f.write((char*)&ns, sizeof(int));
                f.write((char*)layer.data(), ns * sizeof(int));
            }
        }
    }

    // Load index from file
    void load(const string& path) {
        ifstream f(path, ios::binary);
        if (!f) return;
        int n; f.read((char*)&n, sizeof(int));
        f.read((char*)&maxLayer, sizeof(int));
        f.read((char*)&entryPoint, sizeof(int));
        nodes.resize(n);
        for (auto& node : nodes) {
            int vs; f.read((char*)&vs, sizeof(int));
            node.vec.resize(vs);
            f.read((char*)node.vec.data(), vs * sizeof(float));
            int ts; f.read((char*)&ts, sizeof(int));
            node.text.resize(ts);
            f.read(&node.text[0], ts);
            int ls; f.read((char*)&ls, sizeof(int));
            node.neighbors.resize(ls);
            for (auto& layer : node.neighbors) {
                int ns; f.read((char*)&ns, sizeof(int));
                layer.resize(ns);
                f.read((char*)layer.data(), ns * sizeof(int));
            }
        }
    }
};

// CLI interface — Python calls this as a subprocess
int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    string cmd = argv[1];
    HNSW hnsw;
    hnsw.load("index.bin");

    if (cmd == "insert" && argc == 4) {
        // argv[2] = text, argv[3] = comma-separated floats
        string text = argv[2];
        stringstream ss(argv[3]);
        vector<float> vec;
        string val;
        while (getline(ss, val, ',')) vec.push_back(stof(val));
        hnsw.insert(vec, text);
        hnsw.save("index.bin");
        cout << "ok" << endl;

    } else if (cmd == "search" && argc == 3) {
        // argv[2] = comma-separated floats
        stringstream ss(argv[2]);
        vector<float> vec;
        string val;
        while (getline(ss, val, ',')) vec.push_back(stof(val));
        auto results = hnsw.search(vec, 5);
        for (auto& [dist, text] : results) {
            cout << dist << "\t" << text << "\n";
        }

    } else if (cmd == "reset") {
        hnsw = HNSW();
        hnsw.save("index.bin");
        cout << "ok" << endl;
    }

    return 0;
}