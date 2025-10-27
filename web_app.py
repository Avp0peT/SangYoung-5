from flask import Flask, request, jsonify, render_template, send_file
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from inference import HumanParsingInference
import uuid
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 文件大小限制

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# 创建输出目录
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 初始化模型
inference = None

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_model():
    """初始化模型"""
    global inference
    # 检查是否有训练好的模型
    model_path = 'checkpoints/best_model.pth'
    if os.path.exists(model_path):
        inference = HumanParsingInference(model_path=model_path)
        print("模型加载成功")
    else:
        inference = HumanParsingInference()
        print("使用未训练的模型（请先训练模型以获得更好效果）")

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """上传并处理图像"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # 生成唯一文件名
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{unique_id}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # 保存文件
        file.save(filepath)
        
        try:
            # 处理图像
            result = process_image(filepath, filename)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': f'处理图像时出错: {str(e)}'}), 500
    else:
        return jsonify({'error': '不支持的文件类型'}), 400

def process_image(filepath, original_filename):
    """处理上传的图像"""
    # 创建本次处理的唯一目录
    session_id = original_filename.rsplit('.', 1)[0]
    output_dir = os.path.join(RESULTS_FOLDER, session_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理图像
    result = inference.process_image(
        filepath, 
        output_dir=output_dir,
        visualize=True
    )
    
    # 构建返回结果
    response = {
        'success': True,
        'session_id': session_id,
        'original_image': f'/static/uploads/{original_filename}',
        'result_image': f'/static/results/{session_id}/{session_id}_result.png',
        'mask_image': f'/static/results/{session_id}/{session_id}_mask.png',
        'extracted_parts': {}
    }
    
    # 添加提取的身体部位信息
    parts_dir = os.path.join(output_dir, f'{session_id}_parts')
    if os.path.exists(parts_dir):
        for part_name in os.listdir(parts_dir):
            if part_name.endswith('.png'):
                part_path = f'/static/results/{session_id}/{session_id}_parts/{part_name}'
                response['extracted_parts'][part_name.replace('.png', '')] = part_path
    
    # 移动上传的文件到静态目录以便访问
    static_upload_path = os.path.join('static', 'uploads', original_filename)
    os.makedirs(os.path.dirname(static_upload_path), exist_ok=True)
    os.rename(filepath, static_upload_path)
    
    return response

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API接口：预测图像"""
    if 'image' not in request.files:
        return jsonify({'error': '没有图像文件'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # 读取图像
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': '无法读取图像'}), 400
        
        try:
            # 预测
            pred_mask, _, _ = inference.predict(image)
            
            # 提取身体部位
            parts = {}
            for class_idx, class_name in enumerate(inference.class_names):
                if class_idx == 0:  # 跳过背景
                    continue
                
                part_mask = (pred_mask == class_idx).astype(np.uint8)
                if np.any(part_mask > 0):
                    # 提取该部位
                    part_image = image.copy()
                    part_image[part_mask == 0] = 0
                    
                    # 编码为base64
                    _, part_encoded = cv2.imencode('.png', part_image)
                    part_base64 = part_encoded.tobytes().hex()
                    
                    parts[class_name] = part_base64
            
            # 编码预测掩码
            _, mask_encoded = cv2.imencode('.png', pred_mask)
            mask_base64 = mask_encoded.tobytes().hex()
            
            return jsonify({
                'success': True,
                'mask': mask_base64,
                'parts': parts,
                'class_names': inference.class_names
            })
            
        except Exception as e:
            return jsonify({'error': f'预测时出错: {str(e)}'}), 500
    else:
        return jsonify({'error': '不支持的文件类型'}), 400

@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件服务"""
    return send_file(os.path.join('static', filename))

@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': inference is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # 初始化模型
    init_model()
    
    print("启动人体解析Web服务...")
    print("访问 http://localhost:5000 使用服务")
    print("API端点: /api/predict")
    print("健康检查: /health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
