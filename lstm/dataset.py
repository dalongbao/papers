import urllib3
import os

def download_txt_file(url, save_path):
    # Create a PoolManager instance
    http = urllib3.PoolManager()

    # Send a GET request to the URL
    response = http.request('GET', url)

    # Check if the request was successful
    if response.status == 200:
        # Open the file in write binary mode
        with open(save_path, 'wb') as file:
            # Write the content to the file
            file.write(response.data)
        print(f"File downloaded successfully and saved to {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status}")

# URL of the .txt file you want to download
url = "https://courses.cs.washington.edu/courses/cse163/20wi/files/lectures/L04/bee-movie.txt"

# Path where you want to save the file
save_path = os.path.join(os.getcwd(), "beemovie.txt")

# Call the function to download the file
download_txt_file(url, save_path)
