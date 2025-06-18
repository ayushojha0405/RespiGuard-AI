import torch
import math
import cv2
import numpy as np
import torch.nn as nn
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import datetime


# CUDA check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential() if in_channels == out_channels and stride == 1 else \
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
    


class ImprovedYOLOv5(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedYOLOv5, self).__init__()
        self.num_classes = num_classes

        # Backbone
        self.backbone = nn.Sequential(
            ResidualBlock(3, 32, stride=2), # 208x208
            ResidualBlock(32, 64),
            nn.MaxPool2d(2), # 104x104
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            nn.MaxPool2d(2), # 52x52
            ResidualBlock(256, 512),
            ResidualBlock(512, 1024),
        )

        # Neck
        self.neck = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Detection head
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (num_classes + 5), kernel_size=1) # 5 = (x,y,w,h,obj)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        # Reshape output to [batch_size, grid_size*grid_size, num_classes + 5]
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, height * width, channels)

        return x, (height, width)  # Return grid dimensions too


model = ImprovedYOLOv5(num_classes=14).to(device)


def extract_boxes(grid_predictions, confidence_threshold=0.5):
    """
    Convert grid cell predictions to bounding boxes.

    Args:
        grid_predictions: Tensor of shape [grid_cells, 5+num_classes]
        confidence_threshold: Minimum confidence to keep a detection

    Returns:
        List of dictionaries with format:
        {
            'box': [x1, y1, x2, y2],  # normalized 0-1 coordinates
            'confidence': float,       # objectness score
            'class_id': int,           # predicted class ID
            'class_prob': float        # class probability
        }
    """
    grid_cells = grid_predictions.size(0)
    grid_size = int(math.sqrt(grid_cells))
    num_classes = grid_predictions.size(1) - 5

    # Apply sigmoid to objectness
    objectness = torch.sigmoid(grid_predictions[:, 4])
    confident_mask = objectness > confidence_threshold

    if not confident_mask.any():
        return []

    confident_indices = confident_mask.nonzero(as_tuple=True)[0]
    grid_y = torch.div(confident_indices, grid_size, rounding_mode='floor')
    grid_x = confident_indices % grid_size

    confident_preds = grid_predictions[confident_indices]

    x_center = (torch.sigmoid(confident_preds[:, 0]) + grid_x) / grid_size
    y_center = (torch.sigmoid(confident_preds[:, 1]) + grid_y) / grid_size
    w = torch.exp(confident_preds[:, 2])  # Apply exponential to width
    h = torch.exp(confident_preds[:, 3])  # Apply exponential to height

    x1 = torch.clamp(x_center - w/2, 0, 1)
    y1 = torch.clamp(y_center - h/2, 0, 1)
    x2 = torch.clamp(x_center + w/2, 0, 1)
    y2 = torch.clamp(y_center + h/2, 0, 1)

    class_probs = torch.softmax(confident_preds[:, 5:], dim=1)
    class_ids = torch.argmax(class_probs, dim=1)
    max_class_probs = torch.max(class_probs, dim=1)[0]

    detections = []
    for i in range(len(confident_indices)):
        if w[i] * h[i] > 0:
            detections.append({
                'bbox': [x1[i].item(), y1[i].item(), x2[i].item(), y2[i].item()],
                'confidence': objectness[confident_indices[i]].item(),
                'class_id': class_ids[i].item(),
                'class_prob': max_class_probs[i].item()
            })

    return detections


def predict_disease(model, image_path, device, img_size=416, confidence_threshold=0.5):
    """
    Predict lungs disease from image.

    Args:
        model: Trained model
        image_path: Path to the liver image
        device: Device to run inference on
        img_size: Input image size
        confidence_threshold: Threshold for detection confidence

    Returns:
        List of detections with disease class, confidence, and bounding box
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return []

    original_h, original_w = image.shape[:2]
    image = cv2.resize(image, (img_size, img_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.tensor(image).float().unsqueeze(0).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        predictions, grid_size = model(image_tensor)

    # Convert predictions to boxes
    raw_detections = extract_boxes(predictions[0], confidence_threshold)

    # Convert to output format with class names
    class_names = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 'Nodule-Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']


    detections = []
    for det in raw_detections:
        x1, y1, x2, y2 = det['bbox']
        class_id = det['class_id']
        confidence = det['confidence'] * det['class_prob']  # Combined confidence

        # Convert normalized coordinates to image coordinates
        x1_img = x1 * original_w
        y1_img = y1 * original_h
        x2_img = x2 * original_w
        y2_img = y2 * original_h

        detections.append({
            'disease': class_names[class_id],
            'class_id': class_id,
            'confidence': confidence,
            'bbox': [x1_img, y1_img, x2_img, y2_img]
        })

    return detections


def enhanced_visualize_prediction(image_path, detections, save_path=None):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from scipy.ndimage import gaussian_filter

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Original image with heatmap overlay
    ax1.imshow(image)
    ax1.set_title('Disease Detection Heatmap', fontsize=14)
    ax1.axis('off')

    # Medical annotation view
    ax2.imshow(image)
    ax2.set_title('Region of Interest Analysis', fontsize=14)
    ax2.axis('off')

    # Create heatmap
    heatmap = np.zeros((h, w))

    # Disease-specific colors (using pastel colors for less visual clutter)
    disease_colors = {
        'Aortic enlargement': (0.8, 0.2, 0.2),  # Soft red
        'Atelectasis': (0.2, 0.8, 0.2),         # Soft green
        'Calcification': (0.2, 0.2, 0.8),       # Soft blue
        'Cardiomegaly': (0.8, 0.2, 0.8),        # Soft purple
        'Consolidation': (0.2, 0.8, 0.8),       # Soft cyan
        'ILD': (0.8, 0.6, 0.2),                 # Soft orange
        'Infiltration': (0.6, 0.2, 0.8),        # Soft violet
        'Lung Opacity': (0.2, 0.6, 0.8),        # Soft sky blue
        'Nodule-Mass': (0.8, 0.4, 0.2),         # Soft brown
        'Other lesion': (0.6, 0.8, 0.2),        # Soft lime
        'Pleural effusion': (0.8, 0.6, 0.6),    # Soft pink
        'Pleural thickening': (0.6, 0.6, 0.8),  # Soft lavender
        'Pneumothorax': (1.0, 0.4, 0.4),        # Light red
        'Pulmonary fibrosis': (0.4, 0.8, 0.4)   # Light green
    }

    # Group detections by disease
    grouped_detections = {}
    for det in detections:
        if det['confidence'] < 0.3:  # Filter low confidence
            continue
        disease = det['disease']
        if disease not in grouped_detections:
            grouped_detections[disease] = []
        grouped_detections[disease].append(det)

    # Create overlays for each disease
    for disease, group in grouped_detections.items():
        color = disease_colors.get(disease, (0.5, 0.5, 0.5))

        # Create disease-specific heatmap
        disease_heatmap = np.zeros((h, w))

        for det in group:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            confidence = det['confidence']

            # Calculate center and radius
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = max((x2 - x1), (y2 - y1)) // 2

            # Add circular region to heatmap
            y_grid, x_grid = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
            mask = dist_from_center <= radius
            disease_heatmap[mask] = confidence

        # Apply Gaussian blur to smooth the heatmap
        disease_heatmap = gaussian_filter(disease_heatmap, sigma=30)

        # Add to main heatmap
        heatmap += disease_heatmap

        # Add subtle circle indicator in second view
        if group:
            max_conf_det = max(group, key=lambda x: x['confidence'])
            x1, y1, x2, y2 = [int(coord) for coord in max_conf_det['bbox']]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Calculate a much smaller radius
            width = x2 - x1
            height = y2 - y1
            # Reduce radius to 30% of smaller dimension
            radius = min(width, height) * 0.3
            
            # Draw main annotation circle
            circle = Circle((center_x, center_y), radius,
                          fill=False,
                          color=color,
                          alpha=0.9,
                          linestyle='--',
                          linewidth=2.5)
            ax2.add_patch(circle)

            # Add a focal point marker
            center_dot = Circle((center_x, center_y), 
                          radius=3,
                          fill=True,
                          color=color,
                          alpha=1.0)
            ax2.add_patch(center_dot)

            # Add crosshair lines for better region targeting
            line_length = radius * 0.5
            ax2.plot([center_x - line_length, center_x + line_length], 
                    [center_y, center_y], 
                    color=color, 
                    linestyle=':',
                    linewidth=1.5)
            ax2.plot([center_x, center_x], 
                    [center_y - line_length, center_y + line_length], 
                    color=color, 
                    linestyle=':',
                    linewidth=1.5)

            # Move text label closer to circle
            ax2.text(center_x, center_y - radius - 3,
                    f"{disease}\n{max_conf_det['confidence']:.2f}",
                    color=color,
                    fontsize=9,
                    ha='center',
                    bbox=dict(facecolor='white',
                             edgecolor=color,
                             alpha=0.9,
                             pad=1))

    # Normalize and display heatmap
    heatmap = heatmap / heatmap.max()
    ax1.imshow(heatmap, cmap='hot', alpha=0.4)

    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=color, label=disease,
                                 markersize=8)
                      for disease, color in disease_colors.items()
                      if disease in grouped_detections]

    if legend_elements:
        ax2.legend(handles=legend_elements,
                  loc='center left',
                  bbox_to_anchor=(1, 0.5),
                  fontsize=8,
                  title='Detected Conditions')

    # Add summary text
    summary_text = "Key Findings:\n"
    for disease, group in grouped_detections.items():
        avg_conf = np.mean([d['confidence'] for d in group])
        if avg_conf > 0.5:
            summary_text += f"• {disease}: {avg_conf:.2f}\n"

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    fig.text(0.02, 0.02, summary_text, fontsize=10,
             bbox=props, verticalalignment='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def generate_medical_report(image_path, detections, patient_id="Unknown"):
    import datetime
    from collections import defaultdict

    # Initialize disease counts and confidences
    disease_counts = defaultdict(int)
    disease_confidences = defaultdict(list)
    primary_diagnosis = "No abnormalities detected"
    primary_confidence = 0.0

    # Process detections
    if detections:
        for det in detections:
            disease = det['disease']
            confidence = det['confidence']
            disease_counts[disease] += 1
            disease_confidences[disease].append(confidence)

            # Update primary diagnosis if confidence is higher
            avg_conf = sum(disease_confidences[disease]) / len(disease_confidences[disease])
            if avg_conf > primary_confidence:
                primary_diagnosis = disease
                primary_confidence = avg_conf

    current_date = datetime.datetime.now().strftime("%B %d, %Y")

    # Build report content
    report_content = f"""
{'='*80}
                    CHEST X-RAY REPORT
{'='*80}

PATIENT ID: {patient_id}
EXAMINATION DATE: {current_date}
IMAGE ANALYZED: {os.path.basename(image_path)}

{'='*80}
FINDINGS:
{'='*80}
"""

    # Add findings section
    if disease_counts:
        for disease, count in disease_counts.items():
            avg_conf = sum(disease_confidences[disease]) / len(disease_confidences[disease])
            report_content += f"\n{disease.upper()}:\n"
            report_content += f"- Number of regions: {count}\n"
            report_content += f"- Confidence: {avg_conf:.2f}\n"
            report_content += f"- Location: {get_anatomical_location(disease)}\n"
    else:
        report_content += "No significant findings detected in this chest radiograph.\n"

    # Add impression section
    report_content += f"""
{'='*80}
IMPRESSION:
{'='*80}
"""

    if primary_diagnosis != "No abnormalities detected":
        confidence_level = "high" if primary_confidence > 0.8 else "moderate" if primary_confidence > 0.6 else "low"
        report_content += f"1. {primary_diagnosis.upper()} with {confidence_level} confidence ({primary_confidence:.2f})\n"

        # Add secondary findings
        secondary_count = 2
        for disease, confidences in disease_confidences.items():
            avg_conf = sum(confidences) / len(confidences)
            if disease != primary_diagnosis and avg_conf > 0.4:
                report_content += f"{secondary_count}. {disease.upper()} with lower confidence ({avg_conf:.2f})\n"
                secondary_count += 1
    else:
        report_content += "No significant abnormalities detected in this examination.\n"

    # Add recommendations section
    report_content += f"""
{'='*80}
RECOMMENDATIONS:
{'='*80}
"""

    # Add disease-specific recommendations
    for disease in disease_counts.keys():
        recommendations = get_disease_recommendations(disease)
        report_content += f"\nFor {disease.upper()}:\n"
        for i, rec in enumerate(recommendations, 1):
            report_content += f"{i}. {rec}\n"

    # Add signature
    report_content += f"""
{'='*80}

Report generated by AI Chest X-Ray Analysis System
This is a computer-generated report and requires clinical correlation.
{'='*80}
"""

    return report_content
    
    
def get_disease_recommendations(disease):
    """Helper function to return recommendations for each disease"""
    recommendations = {
        'Aortic enlargement': [
            'Cardiac evaluation recommended',
            'Consider echocardiogram',
            'Monitor blood pressure'
        ],
        'Atelectasis': [
            'Chest physiotherapy may be beneficial',
            'Follow-up imaging to monitor resolution',
            'Evaluate for underlying causes'
        ],
        'Cardiomegaly': [
            'Complete cardiac workup recommended',
            'Evaluate for underlying heart conditions',
            'Consider ECG and echocardiogram',
            'Monitor blood pressure and symptoms'
        ],
        'Consolidation': [
            'Antibiotic therapy may be needed',
            'Follow-up chest X-ray recommended',
            'Monitor respiratory status',
            'Consider sputum culture'
        ],
        'Pleural Effusion': [
            'Evaluate for underlying cause',
            'Consider thoracentesis if clinically indicated',
            'Monitor respiratory status',
            'Follow-up imaging recommended'
        ],
        'Pulmonary Edema': [
            'Assess cardiac function',
            'Consider diuretic therapy',
            'Monitor fluid status',
            'Evaluate renal function'
        ],
        'Pneumothorax': [
            'Urgent chest tube placement may be needed',
            'Monitor oxygen saturation',
            'Serial chest X-rays recommended',
            'Assess for respiratory distress'
        ],
        'Infiltration': [
            'Consider broad-spectrum antibiotics',
            'Monitor temperature and white blood cell count',
            'Follow-up imaging to track resolution',
            'Respiratory support as needed'
        ],
        'Emphysema': [
            'Smoking cessation counseling',
            'Pulmonary rehabilitation may be beneficial',
            'Consider bronchodilator therapy',
            'Monitor pulmonary function'
        ],
        'Mass': [
            'Further imaging with CT recommended',
            'Consider biopsy if clinically indicated',
            'Oncology consultation may be needed',
            'Regular follow-up recommended'
        ]
    }
    return recommendations.get(disease, ['Clinical correlation recommended'])

def get_anatomical_location(disease):
    """Helper function to return anatomical location for each disease"""
    locations = {
        'Aortic enlargement': 'Mediastinum - Aortic region',
        'Atelectasis': 'Dependent portions of lung fields',
        'Calcification': 'Variable locations in lung fields',
        'Cardiomegaly': 'Cardiac silhouette',
        'Consolidation': 'Variable locations in lung fields',
        'ILD': 'Bilateral lung fields',
        'Infiltration': 'Variable locations in lung fields',
        'Lung Opacity': 'Variable locations in lung fields',
        'Nodule-Mass': 'Focal area in lung fields',
        'Other lesion': 'Variable location',
        'Pleural effusion': 'Pleural space',
        'Pleural thickening': 'Pleural surfaces',
        'Pneumothorax': 'Pleural space',
        'Pulmonary fibrosis': 'Bilateral lung fields'
    }
    return locations.get(disease, 'Location not specified')



# Load the trained model once
checkpoint_path = 'weights.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def run_model(image_path, patient_id="Unknown"):
    detections = predict_disease(model, image_path, device)

    # Generate annotated image
    annotated_path = f'temp_images/{patient_id}_report.png'
    enhanced_visualize_prediction(image_path, detections, save_path=annotated_path)

    # Generate textual report
    report = generate_medical_report(image_path, detections, patient_id)

    # Save text file (optional)
    with open(f'temp_images/{patient_id}_report.txt', 'w') as f:
        f.write(report)

    # Generate PDF
    pdf_path = generate_pdf_report(annotated_path, report, patient_id)

    return {
    "text": report,
    "pdfPath": pdf_path,
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}


def generate_pdf_report(image_path, report_text, patient_id="Unknown"):
    from reportlab.platypus import Image as RLImage

    pdf_path = f"temp_images/{patient_id}_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title_style = styles['Heading1']
    title_style.textColor = colors.darkblue
    elements.append(Paragraph(f"Chest X-Ray AI Report", title_style))
    elements.append(Spacer(1, 12))

    # Patient ID and Timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_text = f"<b>Patient ID:</b> {patient_id} <br/><b>Generated:</b> {timestamp}"
    elements.append(Paragraph(meta_text, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Add image (resized to fit page)
    try:
        img = RLImage(image_path, width=5.5 * inch, height=4 * inch)
        elements.append(img)
        elements.append(Spacer(1, 12))
    except Exception as e:
        print("Error loading image into PDF:", e)

    # Add report content
    paragraphs = report_text.strip().split("\n")
    for para in paragraphs:
        if para.strip():
            elements.append(Paragraph(para.strip().replace("  ", "&nbsp;&nbsp;"), styles["Normal"]))
            elements.append(Spacer(1, 6))

    # Build PDF
    doc.build(elements)
    return pdf_path
