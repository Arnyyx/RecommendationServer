from flask import Flask, request, jsonify, render_template_string
from firebase_admin import credentials, firestore, initialize_app
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import json
import logging
from datetime import timedelta, datetime, timezone
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

try:
    firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
    cred = credentials.Certificate(json.loads(firebase_credentials))
    initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Firebase: {str(e)}")
    raise


def get_user_interactions(user_id):
    try:
        interactions_ref = db.collection("users").document(user_id).collection("interactions")
        interactions = interactions_ref.get()
        result = {}
        total_score = 0
        max_score = 0
        min_score = float('inf')

        for interaction in interactions:
            data = interaction.to_dict()
            post_id = data["postId"]
            score = data.get("interactionScore", 0.0)
            result[post_id] = score
            total_score += score
            max_score = max(max_score, score)
            min_score = min(min_score, score)

        avg_score = total_score / len(result) if result else 0
        logger.info(
            f"User {user_id} interactions - Total: {len(result)}, Avg score: {avg_score:.3f}, Max: {max_score:.3f}, Min: {min_score:.3f}")

        # Log top 5 interactions
        top_interactions = sorted(result.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Top interactions for {user_id}: {top_interactions}")

        return result
    except Exception as e:
        logger.error(f"Error fetching interactions for {user_id}: {str(e)}")
        return {}


def get_post_embedding(post_id):
    try:
        post_ref = db.collection("posts").document(post_id)
        post_data = post_ref.get().to_dict() or {}
        caption = post_data.get("caption", "")
        keywords = " ".join(post_data.get("keywords", []))
        combined_text = caption + " " + keywords

        logger.debug(f"Post {post_id} - Caption length: {len(caption)}, Keywords: {len(post_data.get('keywords', []))}")

        return model.encode(combined_text)
    except Exception as e:
        logger.error(f"Error fetching embedding for {post_id}: {str(e)}")
        return np.zeros(384)  # Kích thước embedding của all-MiniLM-L6-v2


def analyze_user_profile(user_id, interactions):
    """Phân tích profile người dùng dựa trên interactions"""
    if not interactions:
        return None

    # Lấy thông tin về các bài post đã tương tác
    interacted_posts = []
    for post_id, score in interactions.items():
        try:
            post_data = db.collection("posts").document(post_id).get().to_dict()
            if post_data:
                interacted_posts.append({
                    'post_id': post_id,
                    'score': score,
                    'keywords': post_data.get('keywords', []),
                    'caption': post_data.get('caption', ''),
                    'owner': post_data.get('postOwnerID', '')
                })
        except Exception as e:
            logger.warning(f"Could not fetch post data for {post_id}: {str(e)}")

    # Phân tích keywords phổ biến
    keyword_scores = {}
    for post in interacted_posts:
        for keyword in post['keywords']:
            keyword_scores[keyword] = keyword_scores.get(keyword, 0) + post['score']

    top_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    logger.info(f"User {user_id} profile - Top keywords: {top_keywords}")
    logger.info(f"User {user_id} profile - Total posts interacted: {len(interacted_posts)}")

    return {
        'top_keywords': top_keywords,
        'total_interactions': len(interacted_posts),
        'avg_interaction_score': sum(interactions.values()) / len(interactions)
    }


def calculate_detailed_similarity(user_embedding, post_embedding, post_id, user_profile):
    """Tính toán similarity chi tiết và log các yếu tố ảnh hưởng"""
    base_similarity = cosine_similarity([user_embedding], [post_embedding])[0][0]

    # Lấy thông tin bài post để phân tích
    try:
        post_data = db.collection("posts").document(post_id).get().to_dict()
        post_keywords = post_data.get('keywords', [])
        post_caption = post_data.get('caption', '')

        # Tính keyword overlap bonus
        keyword_bonus = 0
        if user_profile and user_profile['top_keywords']:
            user_top_keywords = [kw[0] for kw in user_profile['top_keywords'][:5]]
            overlap_keywords = set(post_keywords) & set(user_top_keywords)
            keyword_bonus = len(overlap_keywords) * 0.1

        final_score = base_similarity + keyword_bonus

        logger.debug(f"Post {post_id} scoring breakdown:")
        logger.debug(f"  - Base similarity: {base_similarity:.4f}")
        logger.debug(f"  - Keyword bonus: {keyword_bonus:.4f}")
        logger.debug(f"  - Final score: {final_score:.4f}")
        logger.debug(f"  - Post keywords: {post_keywords}")
        logger.debug(f"  - Caption length: {len(post_caption)}")

        return final_score, {
            'base_similarity': base_similarity,
            'keyword_bonus': keyword_bonus,
            'final_score': final_score,
            'post_keywords': post_keywords,
            'caption_length': len(post_caption)
        }

    except Exception as e:
        logger.warning(f"Could not get detailed scoring for post {post_id}: {str(e)}")
        return base_similarity, {'base_similarity': base_similarity, 'final_score': base_similarity}


def recommend_posts(user_id, limit=10):
    try:
        logger.info(f"Starting recommendation for user {user_id}")

        interactions = get_user_interactions(user_id)
        user_profile = analyze_user_profile(user_id, interactions)

        all_posts = db.collection("posts").get()
        post_ids = []
        post_embeddings = []

        for post in all_posts:
            post_data = post.to_dict()
            if post_data.get("postOwnerID") != user_id:
                post_ids.append(post.id)
                post_embeddings.append(get_post_embedding(post.id))

        logger.info(f"Found {len(post_ids)} candidate posts for user {user_id}")

        if not interactions:
            logger.info(f"No interactions for user {user_id}, returning recent posts")
            recent_posts = db.collection("posts") \
                .where("postOwnerID", "!=", user_id) \
                .order_by("timestamp", direction="DESCENDING") \
                .limit(limit).get()
            return [{"postId": p.id, "score": 1.0, "reason": "Recent post"} for p in recent_posts]

        if not post_embeddings:
            logger.info(f"No posts available for recommendation for user {user_id}")
            return []

        # Tạo user embedding từ các bài đã tương tác
        interacted_embeddings = []
        interaction_weights = []

        for post_id, score in interactions.items():
            embedding = get_post_embedding(post_id)
            interacted_embeddings.append(embedding)
            interaction_weights.append(score)

        # Weighted average của user embedding
        user_embedding = np.average(interacted_embeddings, axis=0, weights=interaction_weights)
        logger.info(f"Created user embedding from {len(interacted_embeddings)} interactions")

        # Tính similarity chi tiết cho từng post
        recommendations = []
        detailed_scores = []

        for idx, post_id in enumerate(post_ids):
            final_score, score_details = calculate_detailed_similarity(
                user_embedding, post_embeddings[idx], post_id, user_profile
            )

            recommendations.append({
                "postId": post_id,
                "score": float(final_score),
                "reason": "Similar to your interests"
            })

            detailed_scores.append({
                'post_id': post_id,
                **score_details
            })

        # Sắp xếp và lấy top recommendations
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:limit]
        top_detailed_scores = sorted(detailed_scores, key=lambda x: x['final_score'], reverse=True)[:limit]

        # Log chi tiết về top recommendations
        logger.info(f"Top {len(recommendations)} recommendations for user {user_id}:")
        for i, (rec, details) in enumerate(zip(recommendations, top_detailed_scores)):
            logger.info(f"  #{i + 1}: Post {rec['postId']} - Score: {rec['score']:.4f}")
            logger.info(f"    Base similarity: {details['base_similarity']:.4f}")
            logger.info(f"    Keyword bonus: {details.get('keyword_bonus', 0):.4f}")
            logger.info(f"    Keywords: {details.get('post_keywords', [])}")

        return recommendations

    except Exception as e:
        logger.error(f"Error in recommend_posts for {user_id}: {str(e)}")
        return []


@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
    <html>
        <head><title>Recommendation Server</title></head>
        <body>
            <h1>Recommendation Server</h1>
            <p>Server is running. Use /recommend/<user_id> to get recommendations.</p>
        </body>
    </html>
    """)


@app.route("/status", methods=["GET"])
def status():
    try:
        db.collection("users").limit(1).get()
        logger.info("Status check: Server and Firebase are operational")
        return jsonify(
            {"status": "running", "firebase_connected": True, "timestamp": datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e), "firebase_connected": False})


@app.route("/recommend/<user_id>", methods=["GET"])
def get_recommendations(user_id):
    try:
        limit = int(request.args.get("limit", 10))
        last_post = request.args.get("last_post", None)
        debug_mode = request.args.get("debug", "false").lower() == "true"

        # Bật debug logging nếu được yêu cầu
        if debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info(
            f"Received request for user {user_id} with limit={limit}, last_post={last_post}, debug={debug_mode}")

        # Kiểm tra cache
        cache_ref = db.collection("recommendations").document(user_id)
        cache = cache_ref.get().to_dict()

        if cache and cache.get("timestamp") > datetime.now(timezone.utc) - timedelta(minutes=1):
            logger.info(f"Using cached recommendations for user {user_id}")
            recommendations = cache.get("post_ids", [])
        else:
            logger.info(f"Generating fresh recommendations for user {user_id}")
            recommendations = recommend_posts(user_id, limit * 2)
            cache_ref.set({"post_ids": recommendations, "timestamp": datetime.now(timezone.utc)})

        # Pagination
        if last_post and any(r["postId"] == last_post for r in recommendations):
            start_idx = next(i for i, r in enumerate(recommendations) if r["postId"] == last_post) + 1
            recommendations = recommendations[start_idx:start_idx + limit]
            logger.info(f"Applied pagination: starting from index {start_idx}")
        else:
            recommendations = recommendations[:limit]

        logger.info(f"Returning {len(recommendations)} recommendations for user {user_id}")

        # Reset log level
        if debug_mode:
            logging.getLogger().setLevel(logging.INFO)

        return jsonify({"posts": recommendations})

    except Exception as e:
        logger.error(f"Error in get_recommendations for user {user_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/user-profile/<user_id>", methods=["GET"])
def get_user_profile(user_id):
    """Endpoint mới để xem profile phân tích của user"""
    try:
        interactions = get_user_interactions(user_id)
        profile = analyze_user_profile(user_id, interactions)

        return jsonify({
            "user_id": user_id,
            "profile": profile,
            "total_interactions": len(interactions)
        })
    except Exception as e:
        logger.error(f"Error getting profile for user {user_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
