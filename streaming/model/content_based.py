import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

# Get the project root directory (2 levels up from this file)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
data_path = os.path.join(project_root, 'data', 'final', 'final_cleaned_books.csv')

books_df = pd.read_csv(data_path)

def pearson_score(ratingsPivot, id1, id2):
    if id1 not in ratingsPivot.index or id2 not in ratingsPivot.index:
        return 0.0

    vec1 = ratingsPivot.loc[id1]
    vec2 = ratingsPivot.loc[id2]
    co_mask = vec1.notna() & vec2.notna()

    if co_mask.sum() < 2:
        return 0.0

    a1 = (vec1[co_mask] - vec1[co_mask].mean()).to_numpy()
    a2 = (vec2[co_mask] - vec2[co_mask].mean()).to_numpy()
    denorminator = np.linalg.norm(a1) * np.linalg.norm(a2)
    if denorminator == 0:
        return 0.0
    return float(np.dot(a1, a2) / denorminator)


def cosine_score(ratingsPivot, id1, id2):
    vec1 = ratingsPivot.loc[id1]
    vec2 = ratingsPivot.loc[id2]
    co_mask = vec1.notna() & vec2.notna()

    if co_mask.sum() < 2:
        return 0.0

    a1 = vec1[co_mask].to_numpy()
    a2 = vec2[co_mask].to_numpy()
    denorminator = np.linalg.norm(a1) * np.linalg.norm(a2)
    if denorminator == 0:
        return 0.0
    return float(np.dot(a1, a2) / denorminator)


def pearson_similarity_vector(v1, v2_matrix):
    v1 = v1.flatten()
    v1_mean = np.mean(v1)
    v1_centered = v1 - v1_mean
    v1_norm = np.linalg.norm(v1_centered)

    v2_means = np.mean(v2_matrix, axis=1, keepdims=True)
    v2_centered = v2_matrix - v2_means
    v2_norms = np.linalg.norm(v2_centered, axis=1)

    denominators = v1_norm * v2_norms
    denominators[denominators == 0] = 1e-9

    correlation = np.dot(v2_centered, v1_centered) / denominators
    return correlation


def get_topK_neighbors(ratingsPivot, target_id, k, similarity_name):
    if similarity_name == 'Pearson':
        similarity_score = [(id, pearson_score(ratingsPivot, target_id, id)) for id in ratingsPivot.index if id != target_id]
    elif similarity_name == 'Cosine':
        similarity_score = [(id, cosine_score(ratingsPivot, target_id, id)) for id in ratingsPivot.index if id != target_id]
    else:
        raise ValueError("similarity_name must be in ['Pearson', 'Cosine]")

    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)[:k]
    return similarity_score

def average_precision_at_k(predictions, true_interactions, k):
    if k <= 0 or len(true_interactions) == 0 or len(predictions) == 0:
        return 0.0
    k_eff = min(k, len(predictions))
    ap_k, relevant = 0.0, 0
    for i in range(k_eff):
        if predictions[i][0] in true_interactions:
            relevant += 1
            ap_k += relevant / (i + 1)
    # common choice: divide by min(k, |relevant set|)
    return ap_k / min(k, len(true_interactions))

def normal_discounted_cumulative_gain_at_k(predictions, true_interactions, k):
    if k <= 0 or len(true_interactions) == 0 or len(predictions) == 0:
        return 0.0
    k_eff = min(k, len(predictions))
    dcg = 0.0
    for i in range(k_eff):
        if predictions[i][0] in true_interactions:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(true_interactions))))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(predictions, true_interactions, k):
    if k <= 0 or len(predictions) == 0:
        return 0.0
    k_eff = min(k, len(predictions))
    top_k = [pred[0] for pred in predictions[:k_eff]]
    return sum(item in true_interactions for item in top_k) / k

def recall_at_k(predictions, true_interactions, k):
    if len(true_interactions) == 0 or len(predictions) == 0:
        return 0.0
    k_eff = min(k, len(predictions))
    top_k = [pred[0] for pred in predictions[:k_eff]]
    return sum(item in true_interactions for item in top_k) / len(true_interactions)


def get_recommendation_cf(ratingsPivot, user_id, k_neighbors, similarity_name):
    topK_neighbors = get_topK_neighbors(ratingsPivot, user_id, k_neighbors, similarity_name)

    total, den = {}, {}
    mean_target = ratingsPivot.loc[user_id].mean(skipna=True)

    for neighbor_id, score in topK_neighbors:
        if neighbor_id not in ratingsPivot.index:
            continue

        neighbor_ratings = ratingsPivot.loc[neighbor_id]
        mean_neighbor = neighbor_ratings.mean(skipna=True)

        # Các item neighbor đã rating
        items = ratingsPivot.loc[neighbor_id][ratingsPivot.loc[neighbor_id].notna()].index.tolist()
        # Các item mà target_id chưa rating
        unseen_items = [it for it in items if pd.isna(ratingsPivot.loc[user_id, it])]
        if not unseen_items:
            continue

        for item in unseen_items:
            total[item] = total.get(item, 0.0) + score * (neighbor_ratings[item] - mean_neighbor)
            den[item]   = den.get(item, 0.0)   + abs(score)

    ranking = []
    for item, num in total.items():
        if den[item] != 0:
            pred_rating = mean_target + (num / den[item])
            ranking.append((item, float(pred_rating)))

    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking


def get_recommendation_cb(train_ratingsDF, desc_feat, user_id, similarity_name):
    user_data = train_ratingsDF[train_ratingsDF['customer_index'] == user_id]
    interacted_idx = user_data['product_index'].tolist()
    user_ratings = user_data['rating'].values.reshape(-1, 1)
    interacted_vecs = desc_feat[interacted_idx]

    user_profile_vec = np.sum(interacted_vecs * user_ratings, axis=0) / np.sum(user_ratings)
    user_profile_vec = user_profile_vec.reshape(1, -1)

    candidate_indices = [product_index for product_index in train_ratingsDF['product_index'].unique() if product_index not in interacted_idx]
    candidate_vecs = desc_feat[candidate_indices]

    if similarity_name == 'Cosine':
        sim_scores = cosine_similarity(user_profile_vec, candidate_vecs).flatten()
    elif similarity_name == 'Pearson':
        sim_scores = pearson_similarity_vector(user_profile_vec, candidate_vecs).flatten()
    else:
        raise ValueError("Similarity Name must be in ['Cosine', 'Pearson']")

    recommendations = sorted(zip(candidate_indices, sim_scores), key=lambda x: x[1], reverse=True)
    return recommendations

# def get_recommendation_cb_bm25(train_ratingsDF, user_id, bm25_corpus):
#     user_data = train_ratingsDF[train_ratingsDF['customer_index'] == user_id]
#     interacted_idx = user_data['product_index'].tolist()
#     description_data = books_df.loc[books_df['product_index'].isin(interacted_idx) & ~books_df['description'].isnull(), 'description'].tolist()
#     if not description_data:
#         return []

#     total_scores = defaultdict(float)

#     for desc in description_data:
#         tokens = tokennize_vn(str(desc)).lower().split()
#         scores = bm25_corpus.get_scores(tokens)
#         for idx, score in enumerate(scores):
#             total_scores[idx] += score

#     recommendations = []
#     check_item = []
#     for idx, score in total_scores.items():
#         real_product_id = index_to_product_indexes.get(idx)

#         # Chỉ thêm vào nếu sách này chưa từng đọc
#         if real_product_id and (real_product_id not in interacted_idx) and (real_product_id not in check_item):
#             check_item.append(real_product_id)
#             recommendations.append((real_product_id, score))

#     recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
#     return recommendations

def get_recommendation_test(train_ratingsDF, test_ratingsDF, K, type_rec, similarity_name=None, text_feat=None, typebook_feat=None, bm25_corpus=None):
    preds = []
    user_ids = test_ratingsDF['customer_index'].tolist()
    ap_list, ndcg_list, precision_list, recall_list = [], [], [], []
    train_ratingsPivot = train_ratingsDF.pivot(index='customer_index', columns='product_index', values='rating')

    for i in tqdm(range(len(user_ids)), desc='Recommend items for User'):
        if type_rec == 'cf':
          rec_items = get_recommendation_cf(
              train_ratingsPivot,
              user_id=user_ids[i],
              k_neighbors=10,
              similarity_name=similarity_name
          )
        elif type_rec == 'cb':
          rec_items = get_recommendation_cb(
              train_ratingsDF,
              text_feat,
              user_id=user_ids[i],
              similarity_name=similarity_name
          )
        elif type_rec == 'typebook_based':
          rec_items = get_recommendation_cb(
              train_ratingsDF,
              typebook_feat,
              user_id=user_ids[i],
              similarity_name=similarity_name
          )
        # elif type_rec == 'cb_bm25':
        #   rec_items = get_recommendation_cb_bm25(
        #       train_ratingsDF,
        #       user_id=user_ids[i],
        #       bm25_corpus=bm25_corpus
        #   )
        else:
          raise ValueError(f"Type Recommendation system must be in ['cf', 'cb', 'cb_bm25', 'typebook_based']")

        preds.append(rec_items)
        true_interactions = test_ratingsDF[test_ratingsDF['customer_index'] == user_ids[i]]['product_index'].tolist()

        # Tính các metrics cho mỗi dự đoán
        ap_list.append(average_precision_at_k(rec_items, true_interactions, K))
        ndcg_list.append(normal_discounted_cumulative_gain_at_k(rec_items, true_interactions, K))
        precision_list.append(precision_at_k(rec_items, true_interactions, K))
        recall_list.append(recall_at_k(rec_items, true_interactions, K))

    # Tính giá trị trung bình của tất cả các metrics
    mean_ap = sum(ap_list) / len(ap_list)
    mean_ndcg = sum(ndcg_list) / len(ndcg_list)
    mean_precision = sum(precision_list) / len(precision_list)
    mean_recall = sum(recall_list) / len(recall_list)

    print(f"MAP@{K}: {mean_ap}")
    print(f"NDCG@{K}: {mean_ndcg}")
    print(f"Precision@{K}: {mean_precision}")
    print(f"Recall@{K}: {mean_recall}")

    return preds

def get_recommendation_inference_cb(train_ratingsDF, desc_feat, user_id, similarity_name):
    user_data = train_ratingsDF[train_ratingsDF['customer_index'] == user_id]
    interacted_idx = user_data['product_index'].tolist()
    # print(interacted_idx)
    # user_ratings = user_data['rating'].values.reshape(-1, 1)
    interacted_vecs = desc_feat[interacted_idx]

    user_profile_vec = np.mean(interacted_vecs,axis=0).reshape(1, -1)

    candidate_indices = [product_index for product_index in books_df['product_index'].unique() if product_index not in interacted_idx]
    candidate_vecs = desc_feat[candidate_indices]

    if similarity_name == 'Cosine':
        sim_scores = cosine_similarity(user_profile_vec, candidate_vecs).flatten()
    elif similarity_name == 'Pearson':
        sim_scores = pearson_similarity_vector(user_profile_vec, candidate_vecs).flatten()
    else:
        raise ValueError("Similarity Name must be in ['Cosine', 'Pearson']")

    recommendations = sorted(zip(candidate_indices, sim_scores), key=lambda x: x[1], reverse=True)
    return recommendations