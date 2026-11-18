import networkx as nx


def reconstruct_route(graph: nx.MultiDiGraph, decoded_arcs: list[int]):
    """
    Reconstruct a continuous road route from decoded candidate arcs.

    Intermediate arcs are inserted using shortest directed paths.
    """

    arc_edges = {}

    for u, v, data in graph.edges(data=True):
        arc_edges[data["arc_id"]] = (u, v)

    if not decoded_arcs:
        raise ValueError("Decoded route is empty.")

    for arc_id in decoded_arcs:
        if arc_id not in arc_edges:
            raise ValueError(f"Unknown arc ID: {arc_id}")

    full_arcs = [decoded_arcs[0]]

    for previous_arc, current_arc in zip(
        decoded_arcs, decoded_arcs[1:]
    ):
        if previous_arc == current_arc:
            continue

        _, previous_end = arc_edges[previous_arc]
        current_start, _ = arc_edges[current_arc]

        try:
            path = nx.shortest_path(
                graph,
                source=previous_end,
                target=current_start,
            )
        except nx.NetworkXNoPath:
            raise ValueError(
                f"No road path between arcs {previous_arc} "
                f"and {current_arc}"
            )

        for u, v in zip(path, path[1:]):
            edge_data = graph.get_edge_data(u, v)
            matching = next(
                d["arc_id"]
                for d in edge_data.values()
                if d["arc_id"] >= 0
            )
            full_arcs.append(matching)

        full_arcs.append(current_arc)

    full_nodes = [arc_edges[full_arcs[0]][0]]

    for arc_id in full_arcs:
        full_nodes.append(arc_edges[arc_id][1])

    coordinates = [
        (graph.nodes[node]["lon"], graph.nodes[node]["lat"])
        for node in full_nodes
    ]

    return {
        "arc_ids": full_arcs,
        "node_ids": full_nodes,
        "coordinates": coordinates,
    }
