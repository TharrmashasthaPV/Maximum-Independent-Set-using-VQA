def top_k_counts(counts, k):
    """
    A method to return the list of top k keys sorted with 
    respect to counts[key].
    """
    counts_dict = counts.copy()
    top_counts = []
    for i in range(k):
        max_item = max(counts_dict, key=counts_dict.get)
        top_counts.append(max_item)
        del counts_dict[max_item]

    return top_counts