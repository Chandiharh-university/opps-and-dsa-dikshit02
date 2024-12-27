#include <iostream>
#include <vector>
#include <queue>
#include <climits>

using namespace std;

// Function to implement Dijkstra's Algorithm
void dijkstra(int n, int src, const vector<vector<pair<int, int>>>& adj) {
    vector<int> dist(n, INT_MAX); // Distance array to store the shortest distance from source
    dist[src] = 0;

    // Min-Heap: {distance, node}
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, src});

    while (!pq.empty()) {
        int currDist = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (currDist > dist[u]) continue; // Skip if the distance is already updated

        for (const auto& edge : adj[u]) {
            int v = edge.first;
            int weight = edge.second;

            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }

    // Print the shortest distances
    cout << "Shortest distances from node " << src << ":\n";
    for (int i = 0; i < n; ++i) {
        cout << "Node " << i << " : " << (dist[i] == INT_MAX ? -1 : dist[i]) << endl;
    }
}

int main() {
    int n = 6; // Number of nodes
    vector<vector<pair<int, int>>> adj(n);

    // Add edges to the adjacency list (directed graph)
    adj[0].push_back({1, 4});
    adj[0].push_back({2, 4});
    adj[1].push_back({2, 2});
    adj[1].push_back({3, 5});
    adj[2].push_back({3, 8});
    adj[2].push_back({4, 4});
    adj[3].push_back({5, 6});
    adj[4].push_back({3, 1});
    adj[4].push_back({5, 2});

    // Run Dijkstra's algorithm from source node 0
    dijkstra(n, 0, adj);

    return 0;
}
