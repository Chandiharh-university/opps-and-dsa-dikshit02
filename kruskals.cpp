#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Structure to represent an edge
struct Edge {
    int u, v, weight;
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

// Find function for the Disjoint Set Union (DSU)
int findParent(int node, vector<int>& parent) {
    if (node != parent[node]) {
        parent[node] = findParent(parent[node], parent); // Path compression
    }
    return parent[node];
}

// Union function for the DSU
void unionNodes(int u, int v, vector<int>& parent, vector<int>& rank) {
    int pu = findParent(u, parent);
    int pv = findParent(v, parent);

    if (pu != pv) {
        if (rank[pu] < rank[pv]) {
            parent[pu] = pv;
        } else if (rank[pu] > rank[pv]) {
            parent[pv] = pu;
        } else {
            parent[pv] = pu;
            rank[pu]++;
        }
    }
}

// Kruskal's algorithm
void kruskal(int n, vector<Edge>& edges) {
    // Sort edges by weight
    sort(edges.begin(), edges.end());

    vector<int> parent(n);
    vector<int> rank(n, 0);

    // Initialize DSU
    for (int i = 0; i < n; ++i) {
        parent[i] = i;
    }

    vector<Edge> mst; // Store edges in the MST
    int mstWeight = 0;

    for (const Edge& edge : edges) {
        if (findParent(edge.u, parent) != findParent(edge.v, parent)) {
            mst.push_back(edge);
            mstWeight += edge.weight;
            unionNodes(edge.u, edge.v, parent, rank);
        }
    }

    // Print the MST and its weight
    cout << "Edges in the MST:\n";
    for (const Edge& edge : mst) {
        cout << edge.u << " - " << edge.v << " (Weight: " << edge.weight << ")\n";
    }
    cout << "Total weight of MST: " << mstWeight << endl;
}

int main() {
    int n = 5; // Number of nodes
    vector<Edge> edges = {
        {0, 1, 2},
        {0, 3, 6},
        {1, 2, 3},
        {1, 3, 8},
        {1, 4, 5},
        {2, 4, 7}
    };

    // Run Kruskal's algorithm
    kruskal(n, edges);

    return 0;
}
