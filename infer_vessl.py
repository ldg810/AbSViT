import os
import argparse
import streamlit as st
from PIL import Image
import test_simple

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for AbSViT models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)

    parser.add_argument('--resume', type=str,
                        help='path of a pretrained model to use', required=True
                        )

    parser.add_argument('--model', type=str,
                        help='name of a pretrained model to use',
                        default="absvit_tiny_patch8_224_gap",
                        choices=[
                            "absvit_tiny_patch8_224_gap",
                            "absvit_small_patch16_224",
                            "absvit_base_patch16_224.pth"])

    parser.add_argument('--classes', type=int, nargs='+',
                        help='list of classes to see attention',
                        default=[879,200],
                        )
    
    return parser.parse_args()

st.set_page_config(layout="wide")

args = parse_args()
args.image_path = "demo/multi_object_real_879_200.jpg"
args.resume = "https://berkeley.box.com/shared/static/7415yz4d1l5z0ur6x32k35f8y99zgynq.pth"
args.model = "absvit_tiny_patch8_224_gap"
test_simple.test_simple(args)

col1, col2, col3 = st.columns(3)

with col1:
    # st.header("Results")
    input_img = Image.open(args.image_path)
    st.image(input_img)

with col2:
    # st.header("Results")
    att_map_img = Image.open('fig_attention_map.png')
    st.image(att_map_img)

with col3:
    yaml = """name : AbSViT
        resources:
        cluster: aws-apne2-prod1
        accelerators: T4:1
        image: quay.io/vessl-ai/ngc-tensorflow-kernel:22.12-tf1-py3-202301160809
        run:
        - workdir: /root
            command: |
            git clone https://github.com/ldg810/AbSViT.git
  
          cd /AbSViT
            pip install -r requirements.txt
            streamlit run infer_vessl.py --image_path demo/multi_object_real_879_200.jpg --resume https://berkeley.box.com/shared/static/7415yz4d1l5z0ur6x32k35f8y99zgynq.pth --model absvit_tiny_patch8_224_gap

        runtime: 24h
        ports:
            - 8501
        """
    st.markdown(
        f'<p style="font-family:Courier; color:Black; font-size: 20px;">YAML</p>',
        unsafe_allow_html=True,
    )
    st.code(yaml, language="yaml", line_numbers=False)     