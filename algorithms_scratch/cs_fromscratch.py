def cosine_sim(v1,v2):
    dot_prod = sum(a*b for a, b in zip(v1,v2))
    norm_v1 = sum(a**2 for a in v1)**0.5
    norm_v2 = sum(b**2 for b in v2)**0.5
    if norm_v1 * norm_v2 ==0:
        return 0.0
    return dot_prod/(norm_v1*norm_v2)

def cosine_sim_row(query_vector, matrix):
   return [cosine_sim(query_vector, matrix.iloc[i]) for i in range(len(matrix))]