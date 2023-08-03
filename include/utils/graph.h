//
// Created by rozhin on 03.08.2023.
// Copyright (c) 2021-2023 xsdnn. All rights reserved.
//

#ifndef XSDNN_GRAPH_H
#define XSDNN_GRAPH_H

#include <queue>
#include <unordered_set>
#include "../layers/layer.h"

namespace xsdnn {

/*
 * T: node_callback(node&)
 * U: edge_callback(edge&)
 */
template<typename T, typename U>
void dfs_graph_traverse(layer* root, T&& node_callback, U&& edge_callback) {
    std::queue<layer*> q;
    std::unordered_set<layer*> visited;

    q.push(root);

    while(!q.empty()) {
        layer* curr = q.front();
        visited.insert(curr);
        q.pop();

        node_callback(*curr);

        std::vector<edgeptr_t> edge = curr->next();
        for (auto e : edge) {
            if (e != nullptr) edge_callback(*e);
        }

        std::vector<node*> prev_nodes = curr->prev_nodes();
        for (auto n : prev_nodes) {
            layer* l = dynamic_cast<layer*>(n);
            if (visited.find(l) == visited.end()) {
                q.push(l);
            }
        }

        std::vector<node*> next_nodes = curr->next_nodes();
        for (auto n : next_nodes) {
            layer* l = dynamic_cast<layer*>(n);
            if (visited.find(l) == visited.end()) {
                q.push(l);
            }
        }
    }
}

} // xsdnn

#endif //XSDNN_GRAPH_H
