from flask import Flask, request, jsonify, render_template_string
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import json
import logging
from datetime import datetime

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Kết nối Firebase
try:
    cred = credentials.Certificate(json.loads(os.getenv("FIREBASE_CREDENTIALS")))
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Firebase: {str(e)}")
    raise


# Lấy dữ liệu tương tác người dùng
def get_user_interactions(user_id):
    try:
        user_ref = db.collection("users").document(user_id)
        user_data = user_ref.get().to_dict() or {}
        viewed_posts = user_data.get("viewedPosts", [])
        liked_posts = user_data.get("likedPosts", [])
        logger.info(f"User {user_id} interactions: viewed={len(viewed_posts)}, liked={len(liked_posts)}")
        return viewed_posts, liked_posts
    except Exception as e:
        logger.error(f"Error fetching user interactions for {user_id}: {str(e)}")
        return [], []


# Tạo vector đặc trưng từ keywords
def get_post_features(post_id):
    try:
        post_ref = db.collection("posts").document(post_id)
        post_data = post_ref.get().to_dict() or {}
        keywords = post_data.get("keywords", [])
        all_keywords = ["ai", "android", "kotlin", "tech", "music"]
        features = [1 if kw in keywords else 0 for kw in all_keywords]
        logger.debug(f"Post {post_id} features: {features}")
        return features
    except Exception as e:
        logger.error(f"Error fetching post features for {post_id}: {str(e)}")
        return [0] * len(all_keywords)


# Đề xuất bài viết bằng KNN
def recommend_posts(user_id, limit=10):
    try:
        viewed_posts, liked_posts = get_user_interactions(user_id)
        all_posts = db.collection("posts").get()

        # Lọc bài viết không thuộc về user_id
        post_ids = []
        post_features = []
        for post in all_posts:
            post_data = post.to_dict()
            if post_data.get("postOwnerID") != user_id:  # Loại bỏ bài viết của chính người dùng
                post_ids.append(post.id)
                post_features.append(get_post_features(post.id))

        logger.info(f"Total posts available after filtering: {len(post_ids)}")

        if not viewed_posts and not liked_posts:
            logger.info(f"No interactions for user {user_id}, returning recent posts")
            recent_posts = db.collection("posts") \
                .where("postOwnerID", "!=", user_id) \
                .order_by("timestamp", direction="DESCENDING") \
                .limit(limit).get()
            return [post.id for post in recent_posts]

        if not post_features:  # Nếu không còn bài viết nào sau khi lọc
            logger.info(f"No posts available for recommendation after filtering for user {user_id}")
            return []

        n_neighbors = min(limit, len(post_features))
        logger.info(f"Using n_neighbors={n_neighbors} for KNN")

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(post_features)
        user_vector = np.mean([get_post_features(pid) for pid in liked_posts + viewed_posts], axis=0)
        distances, indices = nbrs.kneighbors([user_vector])

        recommended_ids = [post_ids[idx] for idx in indices[0]]
        logger.info(f"Recommended {len(recommended_ids)} posts for user {user_id}")
        return recommended_ids
    except Exception as e:
        logger.error(f"Error in recommend_posts for user {user_id}: {str(e)}")
        return []


# Trang chủ
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


# Kiểm tra trạng thái
@app.route("/status", methods=["GET"])
def status():
    try:
        db.collection("users").limit(1).get()
        logger.info("Status check: Server and Firebase are operational")
        return jsonify({"status": "running", "firebase_connected": True, "timestamp": datetime.utcnow().isoformat()})
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e), "firebase_connected": False})


# API endpoint
@app.route("/recommend/<user_id>", methods=["GET"])
def get_recommendations(user_id):
    try:
        limit = int(request.args.get("limit", 10))
        logger.info(f"Received request for user {user_id} with limit={limit}")
        recommended_post_ids = recommend_posts(user_id, limit)
        response = jsonify({"postIds": recommended_post_ids})
        logger.info(f"Returning {len(recommended_post_ids)} recommendations for user {user_id}")
        return response
    except Exception as e:
        logger.error(f"Error in get_recommendations for user {user_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
