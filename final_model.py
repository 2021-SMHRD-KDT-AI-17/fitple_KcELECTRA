from flask import Flask
import threading
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import mysql.connector
from mysql.connector import Error, pooling

# Flask 앱 인스턴스 생성
app = Flask(__name__)

# MySQL 커넥션 풀 생성
connection_pool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=5,
    host="project-db-cgi.smhrd.com",
    user="wldhz",
    password="126",
    database="wldhz",
    port=3307
)

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("matthewburke/korean_sentiment")
model = AutoModelForSequenceClassification.from_pretrained("matthewburke/korean_sentiment")

# 리뷰 처리 및 데이터베이스 업데이트 메소드
def process_review(text, trainer_email):
    try:
        connection = connection_pool.get_connection()
        if connection.is_connected():
            cursor = connection.cursor()
            # 토큰화
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            # 모델 추론
            outputs = model(**inputs)
            # 예측 클래스 결정
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

            # 긍정적 리뷰일 경우 1점 추가
            update_point = 1 if predicted_class == 1 else -1  

            # 새로운 리뷰에 대해서만 강사의 평가 점수 업데이트
            cursor.execute("UPDATE fit_trainer SET trainer_point = trainer_point + %s WHERE trainer_email = %s", (update_point, trainer_email))
            connection.commit()
            print(f"Updated trainer_point for {trainer_email} by {update_point} points")

            cursor.close()
            connection.close()
            return True
    except Error as e:
        print(f"Failed to process review: {e}")
        if connection.is_connected():
            cursor.close()
            connection.close()
        return False

def update_reviews():
    while True:
        try:
            connection = connection_pool.get_connection()
            if connection.is_connected():
                cursor = connection.cursor()
                
                cursor.execute("SELECT trainer_email, trainer_review_text FROM fit_trainer_review WHERE is_processed = false")
                reviews = cursor.fetchall()
                

                for trainer_email, text in reviews:
                    print(f"Processing review for {trainer_email}: {text}")
                    if process_review(text, trainer_email):  # 리뷰 처리 및 데이터베이스 업데이트
                        # 리뷰가 처리되었으므로 is_processed 값을 true로 업데이트
                        cursor.execute("UPDATE fit_trainer_review SET is_processed = true WHERE trainer_email = %s AND trainer_review_text = %s", (trainer_email, text))
                        connection.commit()
                        print(f"Updated is_processed for {trainer_email} and review: {text}")

                cursor.close()
                connection.close()
        except Error as e:
            print(f"Exception occurred: {e}")
            if connection.is_connected():
                cursor.close()
                connection.close()

        
        time.sleep(5)  # 5초마다 리뷰를 갱신

def start_update_thread():
    update_thread = threading.Thread(target=update_reviews, daemon=True)
    update_thread.start()
    print("Started update_reviews thread")


def process_gym_review(text):
    try:
        connection = connection_pool.get_connection()
        if connection.is_connected():
            cursor = connection.cursor()
            # 토큰화
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            # 모델 추론
            outputs = model(**inputs)
            # 예측 클래스 결정
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

            # 긍정적 리뷰일 경우 1점 추가, 부정적 리뷰일 경우 -1점 추가
            update_point = 1 if predicted_class == 1 else -1  

            # gym_point 필드 업데이트
            cursor.execute("UPDATE fit_gym SET gym_point = gym_point + %s", (update_point,))
            connection.commit()
            print(f"Updated gym_point by {update_point} points")

            cursor.close()
            connection.close()
            return True
    except Error as e:
        print(f"Failed to process gym review: {e}")
        if connection.is_connected():
            cursor.close()
            connection.close()
        return False

def update_gym_reviews():
    while True:
        try:
            connection = connection_pool.get_connection()
            if connection.is_connected():
                cursor = connection.cursor()
                
                cursor.execute("SELECT gym_review_text FROM fit_gym_review WHERE is_processed = false")
                reviews = cursor.fetchall()
                

                for text, in reviews:
                    print(f"Processing gym review: {text}")
                    if process_gym_review(text):  # gym 리뷰 처리 및 데이터베이스 업데이트
                        # 리뷰가 처리되었으므로 is_processed 값을 true로 업데이트
                        cursor.execute("UPDATE fit_gym_review SET is_processed = true WHERE gym_review_text = %s", (text,))
                        connection.commit()
                        print(f"Updated is_processed for gym review: {text}")

                cursor.close()
                connection.close()
        except Error as e:
            print(f"Exception occurred while processing gym reviews: {e}")
            if connection.is_connected():
                cursor.close()
                connection.close()

        
        time.sleep(5)  # 5초마다 gym 리뷰를 갱신

# 서버 시작 시 gym 리뷰 갱신 스레드 시작
def start_update_gym_thread():
    update_gym_thread = threading.Thread(target=update_gym_reviews, daemon=True)
    update_gym_thread.start()
    print("Started update_gym_reviews thread")

# 서버 시작 시 리뷰 갱신 스레드와 gym 리뷰 갱신 스레드 시작
if __name__ == '__main__':
    print("Starting Flask server...")
    start_update_gym_thread()
    start_update_thread()  # 수정된 부분: 새로운 함수를 시작하기 위해 호출
    app.run(debug=True, host='0.0.0.0', port=5000)
