# 🌱 Smart Crop Defender

Smart Crop Defender is an AI-powered crop health monitoring system designed to detect plant diseases from images captured at scale.  
The project is envisioned to work with **drone-based image acquisition**, making it flexible for **individual farmers as well as large agricultural businesses**.

By combining computer vision and deep learning, the system enables **early disease detection**, helping reduce crop loss and improve yield management.


---


## 🚀 Key Idea
- 📸 Images can be captured **manually or using drones**
- 🚁 Drones enable large-area field scanning efficiently
- 🌾 Suitable for small farms, plantations, and agri-enterprises
- 📊 Can be extended to analytics dashboards and decision systems


---


## 🚀 Features
- Plant disease detection using deep learning (CNN)
- Supports image input from drones or cameras
- Trained model saved in `.keras` format
- Scalable design for future automation
- Ready for web or API-based integration


---


## 🧠 Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Convolutional Neural Networks (CNN)


---


## 📂 Project Structure
Smart_Crop_Defender/<br>
│──  Money Plant Diseases Dataset/<br>
   &emsp;│── Healthy<br>
   &emsp;│── Bacterial Disease<br>
   &emsp;│── Manganese Toxicity<br>
│── Plant_disease_model_3.keras<br>
│── train_3.py<br>
│── test_2.py<br>
│── app_2.py<br>
│── templatesbr>


---


## ⚙️ How It Works
1. Crop images are captured (camera or drone-mounted camera)
2. Images are preprocessed and labeled
3. CNN model is trained using Keras
4. Trained model predicts disease presence on new images
5. Results can guide early intervention and treatment


---


## ▶️ How to Run
```bash
pip install tensorflow numpy opencv-python
python app_2.py
```


---


## 📌 Use Cases
- Drone-based crop health monitoring
- Early disease detection for farmers
- Large-scale farm and plantation analysis
- Smart agriculture and precision farming
- AI-assisted agri-business decision systems


---


## 🔮 Future Enhancements
- Integration with live drone feeds
- Support for multiple crops and diseases
- Web & mobile dashboards for farmers
- GPS-based field mapping and reports
- Real-time alerts and analytics
