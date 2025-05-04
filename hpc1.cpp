#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
#include <chrono>  // Include for chrono timing

using namespace std;
using namespace std::chrono;

// Graph class definition
class Graph {
    int V;  // Number of vertices
    vector<vector<int>> adj;  // Adjacency list

public:
    // Constructor to initialize graph with V vertices
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    // Add an undirected edge between vertex u and vertex v
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Parallel Breadth-First Search (BFS)
    void parallelBFS(int start) {
        vector<bool> visited(V, false);  // Track visited nodes
        vector<int> currentLevel;        // Nodes in the current level
        currentLevel.push_back(start);
        visited[start] = true;

        while (!currentLevel.empty()) {
            vector<int> nextLevel;  // Nodes to be visited in the next level

            // Parallel loop for processing all nodes at the current level
            #pragma omp parallel for
            for (int i = 0; i < currentLevel.size(); i++) {
                int u = currentLevel[i];
                cout << u << " ";

                // Explore all adjacent nodes
                for (int v : adj[u]) {
                    if (!visited[v]) {
                        // Critical section to avoid race conditions
                        #pragma omp critical
                        {
                            if (!visited[v]) {
                                visited[v] = true;
                                nextLevel.push_back(v);
                            }
                        }
                    }
                }
            }
            // Move to the next level
            currentLevel = nextLevel;
        }
        cout << endl;
    }

    // Serial Breadth-First Search (BFS)
    void serialBFS(int start) {
        vector<bool> visited(V, false);  // Track visited nodes
        queue<int> q;                    // Queue for BFS traversal
        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            cout << u << " ";

            // Explore all adjacent nodes
            for (int v : adj[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }
        cout << endl;
    }

    // Parallel Depth-First Search (DFS)
    void parallelDFS(int start) {
        vector<bool> visited(V, false);  // Track visited nodes
        stack<int> s;                    // Stack for DFS traversal
        s.push(start);

        while (!s.empty()) {
            int u;

            // Critical section to safely pop from the stack
            #pragma omp critical
            {
                if (!s.empty()) {
                    u = s.top();
                    s.pop();
                } else {
                    u = -1; // Invalid node
                }
            }

            if (u != -1 && !visited[u]) {
                visited[u] = true;
                cout << u << " ";

                // Parallel loop for exploring adjacent nodes
                #pragma omp parallel for
                for (int i = 0; i < adj[u].size(); i++) {
                    int v = adj[u][i];
                    if (!visited[v]) {
                        // Push to stack in a critical section
                        #pragma omp critical
                        s.push(v);
                    }
                }
            }
        }
        cout << endl;
    }

    // Serial Depth-First Search (DFS)
    void serialDFS(int start) {
        vector<bool> visited(V, false);  // Track visited nodes
        stack<int> s;                    // Stack for DFS traversal
        s.push(start);

        while (!s.empty()) {
            int u = s.top();
            s.pop();

            if (!visited[u]) {
                visited[u] = true;
                cout << u << " ";

                // Explore all adjacent nodes
                for (int v : adj[u]) {
                    if (!visited[v]) {
                        s.push(v);
                    }
                }
            }
        }
        cout << endl;
    }
};


// int main() {
//     int V;
//     cout << "Enter the number of vertices: ";
//     cin >> V;

//     Graph g(V);

//     int edgeCount;
//     cout << "Enter the number of edges: ";
//     cin >> edgeCount;

//     cout << "Enter the edges (in format 'source destination'): \n";
//     for (int i = 0; i < edgeCount; i++) {
//         int u, v;
//         cin >> u >> v;
//         g.addEdge(u, v);
//     }

//     // Parallel BFS
//     double start_time = omp_get_wtime();
//     cout << "Parallel BFS traversal starting from node 0: ";
//     g.parallelBFS(0);
//     double end_time = omp_get_wtime();
//     cout << "Parallel BFS time: " << (end_time - start_time) * 1000 << " ms\n";

//     // Serial BFS
//     start_time = omp_get_wtime();
//     cout << "Serial BFS traversal starting from node 0: ";
//     g.serialBFS(0);
//     end_time = omp_get_wtime();
//     cout << "Serial BFS time: " << (end_time - start_time) * 1000 << " ms\n";

//     // Parallel DFS
//     start_time = omp_get_wtime();
//     cout << "Parallel DFS traversal starting from node 0: ";
//     g.parallelDFS(0);
//     end_time = omp_get_wtime();
//     cout << "Parallel DFS time: " << (end_time - start_time) * 1000 << " ms\n";

//     // Serial DFS
//     start_time = omp_get_wtime();
//     cout << "Serial DFS traversal starting from node 0: ";
//     g.serialDFS(0);
//     end_time = omp_get_wtime();
//     cout << "Serial DFS time: " << (end_time - start_time) * 1000 << " ms\n";

//     return 0;
// }


int main() {
    int V;
    cout << "Enter the number of vertices: ";
    cin >> V;

    Graph g(V);

    int edgeCount;
    cout << "Enter the number of edges: ";
    cin >> edgeCount;

    cout << "Enter the edges (in format 'source destination'): \n";
    for (int i = 0; i < edgeCount; i++) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    // Parallel BFS
    auto start_time = high_resolution_clock::now();
    cout << "Parallel BFS traversal starting from node 0: ";
    g.parallelBFS(0);
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time).count();
    cout << "Parallel BFS time: " << duration << " ms\n";

    // Serial BFS
    start_time = high_resolution_clock::now();
    cout << "Serial BFS traversal starting from node 0: ";
    g.serialBFS(0);
    end_time = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end_time - start_time).count();
    cout << "Serial BFS time: " << duration << " ms\n";

    // Parallel DFS
    start_time = high_resolution_clock::now();
    cout << "Parallel DFS traversal starting from node 0: ";
    g.parallelDFS(0);
    end_time = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end_time - start_time).count();
    cout << "Parallel DFS time: " << duration << " ms\n";

    // Serial DFS
    start_time = high_resolution_clock::now();
    cout << "Serial DFS traversal starting from node 0: ";
    g.serialDFS(0);
    end_time = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end_time - start_time).count();
    cout << "Serial DFS time: " << duration << " ms\n";

    return 0;
}