import requests

def download_file(url, save_path):
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    with open(save_path, 'wb') as file:
        file.write(response.content)
    print(f'Downloaded {save_path}')

def main():
    # URLs of the model files
    prototxt_url = 'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt'
    model_url = 'https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel'
    
    # Paths to save the downloaded files
    prototxt_path = 'deploy.prototxt'
    model_path = 'mobilenet_iter_73000.caffemodel'
    
    # Download the files
    download_file(prototxt_url, prototxt_path)
    download_file(model_url, model_path)

if __name__ == "__main__":
    main()
