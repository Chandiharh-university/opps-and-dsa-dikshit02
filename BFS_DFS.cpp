#include <iostream>
#include <vector>
#include <queue>

using namespace std;

void bfs(int start, const vector<vector<int>>& adj, vector<bool>& visited) {
    queue<int> q;
    q.push(start);
    visited[start] = true;

    cout << "BFS Traversal: ";
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        cout << node << " ";

        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
    cout << endl;
}

void dfs(int node, const vector<vector<int>>& adj, vector<bool>& visited) {
    visited[node] = true;
    cout << node << " ";

    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            dfs(neighbor, adj, visited);
        }
    }
}

int main() {
    int n = 5; // Number of nodes
    vector<vector<int>> adj = {
        {},           // Node 0 (no edges)
        {2, 3},       // Node 1
        {1, 4},       // Node 2
        {1, 5},       // Node 3
        {2},          // Node 4
        {3}           // Node 5
    };

    vector<bool> visited(n + 1, false);

    // Perform BFS
    bfs(1, adj, visited);

    // Reset visited for DFS
    fill(visited.begin(), visited.end(), false);

    // Perform DFS
    cout << "DFS Traversal: ";
    dfs(1, adj, visited);
    cout << endl;

    return 0;
}
