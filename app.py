from flask import Flask, request, jsonify, render_template_string
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import json

app = Flask(__name__)

# Kết nối Firebase từ biến môi trường
cred = credentials.Certificate(json.loads(os.getenv("FIREBASE_CREDENTIALS")))
firebase_admin.initialize_app(cred)
db = firestore.client()


# Lấy dữ liệu tương tác người dùng
def get_user_interactions(user_id):
    user_ref = db.collection("users").document(user_id)
    user_data = user_ref.get().to_dict() or {}
    return user_data.get("viewedPosts", []), user_data.get("likedPosts", [])


# Tạo vector đặc trưng từ keywords
def get_post_features(post_id):
    post_ref = db.collection("posts").document(post_id)
    post_data = post_ref.get().to_dict() or {}
    keywords = post_data.get("keywords", [])
    all_keywords = ["ai", "android", "kotlin", "tech", "music"]  # Danh sách từ khóa phổ biến
    return [1 if kw in keywords else 0 for kw in all_keywords]


# Đề xuất bài viết bằng KNN
def recommend_posts(user_id, limit=10):
    viewed_posts, liked_posts = get_user_interactions(user_id)
    all_posts = db.collection("posts").get()

    post_ids = [post.id for post in all_posts]
    post_features = [get_post_features(post.id) for post in all_posts]

    if not viewed_posts and not liked_posts:
        recent_posts = db.collection("posts").order_by("timestamp", direction="DESCENDING").limit(limit).get()
        return [post.id for post in recent_posts]

    nbrs = NearestNeighbors(n_neighbors=limit, algorithm="auto").fit(post_features)
    user_vector = np.mean([get_post_features(pid) for pid in liked_posts + viewed_posts], axis=0)
    distances, indices = nbrs.kneighbors([user_vector])

    return [post_ids[idx] for idx in indices[0]]


# API endpoint
@app.route("/recommend/<user_id>", methods=["GET"])
def get_recommendations(user_id):
    limit = int(request.args.get("limit", 10))
    recommended_post_ids = recommend_posts(user_id, limit)
    return jsonify({"postIds": recommended_post_ids})


@app.route("/", methods=["GET"])
def home():
    html_content = """
    <html>
        <head>
            <title>Recommendation Server</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
                h1 { color: #333; }
                p { font-size: 18px; }
            </style>
        </head>
        <body>
            <h1>Welcome to the Recommendation Server</h1>
            <p>This server is running and ready to provide post recommendations.</p>
            <p>Use the endpoint: <code>/recommend/&lt;user_id&gt;?limit=&lt;number&gt;</code></p>
        </body>
    </html>
    """
    return render_template_string(html_content)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
