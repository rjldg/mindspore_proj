import os
import boto3
import numpy as np
import mindspore as ms
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from mindspore import context, Tensor
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.dataset.transforms import vision
from mindspore.nn import Softmax, Sigmoid
from PIL import Image
from image_classifier.resnet50_archi import resnet50

load_dotenv()

# FOR RESNET-50 IMAGE CLASSIFIER
cfg = {
    'HEIGHT': 224,
    'WIDTH': 224,
    '_R_MEAN': 123.68,
    '_G_MEAN': 116.78,
    '_B_MEAN': 103.94,
    '_R_STD': 1,
    '_G_STD': 1,
    '_B_STD': 1,
    'num_class': 8, 
    'model_path': 'C:/Users/Rance/LocalRepositories/mindspore_proj/image_classifier/model_resnet/resnet-ai_2-10_21_86.8.ckpt'
}

# FOR RAG-BASED FOUNDATIONAL LLM
client = boto3.client('bedrock-agent-runtime',
                      region_name = os.getenv("AI_SPEC_REGION"),
                      aws_access_key_id = os.getenv("AI_SPEC_ACCESS_KEY_ID"),
                      aws_secret_access_key = os.getenv("AI_SPEC_SECRET_ACCESS_KEY_ID"))
class_names = {0:'algal_leaf',1:'anthracnose',2:'bird_eye_spot',3:'brown_blight',4:'gray_light',5:'healthy',6:'red_leaf_spot',7:'white_spot'}
knowledge_base_id = os.getenv("AI_SPEC_KNOWLEDGE_BASE_ID")
model_arn = os.getenv("AI_SPEC_MODEL_ARN")

def preprocess_image(image_path):
    
    image = Image.open(image_path).convert('RGB')
    image = image.resize((cfg['WIDTH'], cfg['HEIGHT']))
    image = np.array(image).astype(np.float32)
    
    # Normalize the image
    image = (image - [cfg['_R_MEAN'], cfg['_G_MEAN'], cfg['_B_MEAN']]) / [cfg['_R_STD'], cfg['_G_STD'], cfg['_B_STD']]
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0) 
    
    return Tensor(image, ms.float32)

def load_model():

    net = resnet50(class_num=cfg['num_class'])
    
    param_dict = load_checkpoint(cfg['model_path'])
    load_param_into_net(net, param_dict)
    
    model = Model(net)
    
    return model

def predict(image_path):

    image = preprocess_image(image_path)
    model = load_model()

    # Predict and calculate confidence level
    output = model.predict(image)
    softmax = Softmax()
    probabilities = softmax(output).asnumpy()
    predicted_class = np.argmax(probabilities, axis=1)[0]
    
    return class_names[predicted_class], probabilities[0][predicted_class]

def retrieve_and_generate(input, knowledge_base_id, model_arn):
    response = client.retrieve_and_generate(
        input = {
            'text': input
        },
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': knowledge_base_id,
                'modelArn': model_arn
            }
        }
    )

    return response

def image_classifier(image_path):

    predicted_class, confidence = predict(image_path)
    
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

    # Dispaly the results accordingly
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
    plt.axis('off')
    plt.show()



    
if __name__ == '__main__':
    print("******************************************************** TEA LEAF DISEASE CLASSIFICATION ********************************************************\n");
    image_path = input("\tEnter the filename/path of image: ")
    if os.path.exists(image_path):
        image_classifier(image_path)
        print("OK")
    else:
        print(f"Error: File '{image_path}' does not exist.")