# Vehicle Tracking and Counting
This project is a real-time vehicle counter application that utilizes the YOLO object detection model and a custom tracking algorithm. It supports real-time detection via webcam and video file input, enabling users to count and classify vehicles such as motorcycles, cars, buses, and trucks.

## Installation Instructions
- Download the model provided in vehicle detection repository.
- Ensure you have Python installed. Then, install the required libraries
- Execute the application via the command line/terminal (see usage instructions below).

## Usage
Run the application with the following command:
python main.py --input "0" --output "output.mp4" --model_path "path/to/your/model.pt"
Notes: 
- Use "0" as the --input to open the webcam for real-time vehicle detection and counting.
- Replace "0" with the file path to a video to count vehicles from the video file.
- Specify the path to your YOLO model using the --model_path argument.

## Example Output
You can see the output video through this link https://drive.google.com/file/d/1f3tUvj7vKtoEJsuhdrVV3tnsgihIgVz6/view?usp=sharing
