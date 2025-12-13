# ASL Gen: American Sign Language Hand Pose Generation/Detection

## Installation 

1. Clone the repo
    ```
    git clone https://github.com/hungdche/CS441_ASLGen.git
    cd CS441_ASLGen
    ```

2. Create conda env
    ```
    conda create -n "aslgen" python=3.10.13
    conda activate aslgen
    pip install -r requirements.txt
    ```

3. **[Optional]** Install SAM2 for postprocessing 
    ```
    git clone https://github.com/facebookresearch/sam2.git && cd sam2
    pip install -e .

    # download the checkpoints
    cd checkpoints && \
    ./download_ckpts.sh && \
    cd ..
    ```

## Data capturing 

1. Capture the images given the target ASL letter
    ```
    python image_capture.py \
        --alphabet <your_desired_letter> \
        --target_resolution <res> \
        --num_images <number_of_images> 
    ```

    Press `s` to save the desired image. After saving `num_images` images, the script will terminate. 

2. **[Optional]** Data Postprocessing. Required to install SAM2 like above
    ```
    python image_postprocessing.py
    ```
    Postprocessed images will be saved in `output/`


## Running the code
All of the code are outlined in `aslgen.ipynb`. Follow through with the code and enjoy. 