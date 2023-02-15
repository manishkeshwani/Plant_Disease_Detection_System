import streamlit as st
import numpy as np
from tensorflow.keras.utils import load_img
import tensorflow as tf
from streamlit_option_menu import option_menu

from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input

with st.sidebar:
    selected = option_menu(menu_title = "Leaf Doctor",
                           
                           options=['Apple','Tomato','Potato','Corn','Cherry','Pepper','Peach','Grapes','Strawberry','Orange','Rice','About'],
                           
                           icons=['app','app','app','app','app','app','app','app','app','app','app','app'],
                           
                           menu_icon = 'cast',
                                                     
                           orientation = 'vertical',
                           
                           default_index = 0)



# funtion to predict the disease from trained model
def predict(model,img,class_names):


    images = load_img(img,target_size =(256,256))
    i = img_to_array(images)
    im = preprocess_input(i) 
    img = np.expand_dims(im , axis = 0)
    # img_array = tf.expand_dims(images,0)
    predictions = model.predict(images)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence



    # images = load_img(img,target_size =(256,256)) 
    # img_array = tf.expand_dims(images,0)
    # predictions = model.predict(img_array)
    # predicted_class = class_names[np.argmax(predictions[0])]
    # confidence = round(100 * (np.max(predictions[0])), 2)
    # return predicted_class, confidence


# Funtion to print image and predicted result
def image_model_prediction(img,model,class_names):
    st.image(img,width=700)
    # MODEL = tf.keras.models.load_model(model, compile=False)
    MODEL = load_model(model, compile=False)
    if st.button('Predict'):      
      classs , comfidence = predict(MODEL,img,class_names)
      if(classs != "Healthy"):
        st.success(f"The leaf Disease is {classs} which is {comfidence}% confirmed")
      else:
        st.success(f"The leaf is {classs} {comfidence}% confirmed")

    

# Apple Page
if(selected == 'Apple'):
    st.title("Apple Disease Detection")
    st.markdown('---')
    apple = st.file_uploader("Please Upload an Image",type=['jpg','png','jpeg'],key="1")
    class_names = {0:'Apple__Scab', 
                    1:'Black_rot', 
                    2:'Cedar_apple_rust', 
                    3:'Healthy'}

    if apple is not None:
      image_model_prediction(apple,"apple_model.h5",class_names)
        

# Tomato Page
if(selected == 'Tomato'):
    st.title("Tomato Disease Detection")
    st.markdown('---')
    tomato = st.file_uploader("Please Upload an Image",type=['jpg','png','jpeg'],key="2")
    class_names = {0:'Tomato Bacterial_spot',
                    1: 'Tomato Early_blight',
                    2: 'Tomato Late_blight', 
                    3:'Tomato Leaf_Mold', 
                    4:'Tomato Septoria_leaf_spot',
                    5: 'Tomato Spider_mites Two-spotted_spider_mite', 
                    6:'Tomato Target_Spot',
                    7: 'Tomato_Yellow_Leaf_Curl_Virus',
                    8: 'Tomato_mosaic_virus',
                    9: 'Healthy'}
    if tomato is not None:
      image_model_prediction(tomato,"tomato_model.h5",class_names)


# Potato Page
if(selected == 'Potato'):
    st.title("Potato Disease Detection")
    st.markdown('---')
    potato = st.file_uploader("Please Upload an Image",type=['jpg','png','jpeg'],key="3")
    class_names = {0 : 'Early_Blight',
                    1: 'Healthy',
                    2: 'Late_Blight'}
    if potato is not None:
      image_model_prediction(potato,"potato_model.h5",class_names)


# Corn Page
if(selected == 'Corn'):
    st.title("Corn Disease Detection")
    st.markdown('---')
    corn = st.file_uploader("Please Upload an Image",type=['jpg','png','jpeg'],key="4")
    class_names = {0:'Blight',
                  1:'Common_Rust',
                  2:'Gray_Leaf_Spot',
                  3:'Healthy'}
    if corn is not None:
        image_model_prediction(corn,"corn_model.h5",class_names)


# Cherry Page
if(selected == 'Cherry'):
    st.title("Cherry Disease Detection")
    st.markdown('---')
    cherry = st.file_uploader("Please Upload an Image",type=['jpg','png','jpeg'],key="5")
    class_names = {0:'Powdery_mildew', 1:'Healthy'}
    if cherry is not None:
        image_model_prediction(cherry,"cherry_model.h5",class_names)


# Pepper Page
if(selected == 'Pepper'):
    st.title("Pepper Disease Detection")
    st.markdown('---')
    pepper = st.file_uploader("Please Upload an Image",type=['jpg','png','jpeg'],key="6")
    class_names = {0:'Bacterial_spot',1:'Healthy'}
    if pepper is not None:
        image_model_prediction(pepper,"pepper_model.h5",class_names)


# Peach Page
if(selected == 'Peach'):
    st.title("Peach Disease Detection")
    st.markdown('---')
    peach = st.file_uploader("Please Upload an Image",type=['jpg','png','jpeg'],key="7")
    class_names = {0:'Bacterial_spot',1:'Healthy'}
    if peach is not None:
        image_model_prediction(peach,"peach_model.h5",class_names)


# Grapes Page
if(selected == 'Grapes'):
    st.title("Grapes Disease Detection")
    st.markdown('---')
    grapes = st.file_uploader("Please Upload an Image",type=['jpg','png','jpeg'],key="8")
    class_names = {0:'Grape Black_rot',
                    1: 'Grape Esca_(Black_Measles)',
                    2: 'Grape Leaf_blight_(Isariopsis_Leaf_Spot)',
                    3: 'Healthy'}

    if grapes is not None:
      image_model_prediction(grapes,"grapes_model.h5",class_names)


# Strawberry Page
if(selected == 'Strawberry'):
    st.title("Strawberry Disease Detection")
    st.markdown('---')
    strawberry = st.file_uploader("Please Upload an Image",type=['jpg','png','jpeg'],key="9")
    class_names = {0:'Leaf_scorch',1: 'Healthy'}
    if strawberry is not None:
        image_model_prediction(strawberry,"strawberry_model.h5",class_names)


# Orange Page
if(selected == 'Orange'):
    st.title("Orange Disease Detection")
    st.markdown('---')
    orange = st.file_uploader("Please Upload an Image",type=['jpg','png','jpeg'],key="10")
    class_names = { 0: 'Haunglongbing_(Citrus_greening)',
                    1: 'Healthy'}

    if orange is not None:
      image_model_prediction(orange,"orange_model.h5",class_names)


# Rice Page
if(selected == 'Rice'):
    st.title("Rice Disease Detection")
    st.markdown('---')
    rice = st.file_uploader("Please Upload an Image",type=['jpg','png','jpeg'],key="11")
    class_names = {0:'Bacterial leaf blight', 1:'Brown spot', 2:'Leaf smut'}
    if rice is not None:
      image_model_prediction(rice,"rice_model.h5",class_names)


# About Page
if(selected == 'About'):
    st.title("About")
    st.markdown('---')
    st.write("The existing method for plant disease detection is simply naked eye observation by experts through which identification and detection of plant diseases is done. For doing so, a large team of experts as well as continuous monitoring of plant is required, which costs very high when we do with large farms. At the same time, in some countries, farmers do not have proper facilities or even idea that they can contact to experts. Due to which consulting experts even cost high as well as time consuming too. In such conditions, Leaf Doctor proves to be beneficial in monitoring large fields of crops. Automatic detection of the diseases by just seeing the symptoms on the plant leaves makes it easier as well as cheaper.")
    