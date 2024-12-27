#include <iostream>
#include <vector>
#include <queue>
#include <climits>

using namespace std;

// Function to implement Prim's Algorithm
void primsAlgorithm(int n, const vector<vector<pair<int, int>>>& adj) {
    vector<int> key(n, INT_MAX); // Store the minimum weight edge for each node
    vector<int> parent(n, -1);  // Store the parent of each node in the MST
    vector<bool> inMST(n, false); // Mark nodes already included in the MST

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    key[0] = 0;
    pq.push({0, 0}); // {key, node}

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        if (inMST[u]) continue; // Skip if already in MST
        inMST[u] = true;

        for (const auto& edge : adj[u]) { // Replace structured bindings
            int v = edge.first;
            int weight = edge.second;

            if (!inMST[v] && weight < key[v]) {
                key[v] = weight;
                parent[v] = u;
                pq.push({key[v], v});
            }
        }
    }

    // Print the edges in the MST
    cout << "Edges in the MST:\n";
    for (int i = 1; i < n; ++i) {
        cout << parent[i] << " - " << i << "\n";
    }
}

int main() {
    int n = 5; // Number of nodes
    vector<vector<pair<int, int>>> adj(n);

    // Add edges to the adjacency list (undirected graph)
    adj[0].push_back({1, 2});
    adj[1].push_back({0, 2});

    adj[0].push_back({3, 6});
    adj[3].push_back({0, 6});

    adj[1].push_back({2, 3});
    adj[2].push_back({1, 3});

    adj[1].push_back({3, 8});
    adj[3].push_back({1, 8});

    adj[1].push_back({4, 5});
    adj[4].push_back({1, 5});

    adj[2].push_back({4, 7});
    adj[4].push_back({2, 7});

    // Run Prim's algorithm
    primsAlgorithm(n, adj);

    return 0;
}
