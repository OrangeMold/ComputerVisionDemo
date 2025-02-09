from tkinter import *
from tkinter import ttk
from tkinter.ttk import *
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

class SafetyApp:

    window_width = 1080
    window_height = 720


    def __init__(self, window):
        self.window = window
        self.window.title("SafeShift")
        
        # Initialize camera and model
        self.cap = cv2.VideoCapture(0)
        
        # Get actual camera resolution
        self.window_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.window_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set window size to match camera resolution
        self.window.geometry(f"{self.window_width}x{self.window_height}")
        self.window.minsize(480, 480)
        
        # Initialize camera and model
        self.cap = cv2.VideoCapture(0)
        self.model = YOLO("best10epoch.pt")
        
        # Create label for video feed
        self.video_label = Label(window)
        self.video_label.pack(fill=BOTH, expand=True)

        # Create control frame for buttons
        self.control_frame = Frame(window)
        self.control_frame.pack(side=BOTTOM, fill=X, padx=5, pady=5)
        
        # Remove the physical button section and keep the detection_active variable
        self.detection_active = False

        fileOptions = ["Restart", "Settings", "Export Report"]
        viewOptions = ["Change Camera", "Change Model", "Change Resolution", "Toggle Detection"]  # Added Toggle Detection
        menuBar = Menu(self.window)
        file = Menu(menuBar, tearoff=0)
        for i in fileOptions:
            file.add_command(label=i, command=None)

        menuBar.add_cascade(label="File", menu=file)

        View = Menu(menuBar, tearoff=0)
        # Add commands with specific functions for view menu items
        View.add_command(label="Change Camera", command=self.openSettings)
        View.add_command(label="Change Model", command=self.openSettings)
        View.add_command(label="Change Resolution", command=self.openSettings)
        View.add_command(label="Toggle Detection", command=self.toggle_detection)

        file.add_command(label="Settings", command = self.openSettings)
        file.add_command(label="Exit", command=self.window.quit)

        menuBar.add_cascade(label="View", menu=View)

        self.window.geometry("1080x720")
        self.window.config(menu=menuBar)
        
        # Start video stream
        self.update_frame()
    
    def toggle_detection(self):
        self.detection_active = not self.detection_active
        # Update menu item text to reflect current state
        view_menu = self.window.nametowidget(self.window.winfo_parent()).menubar.children['view']
        if self.detection_active:
            view_menu.entryconfigure("Toggle Detection", label="Stop Detection")
        else:
            view_menu.entryconfigure("Toggle Detection", label="Start Detection")
    
    def openSettings(self):
        # Create settings window
        settings_window = Toplevel(self.window)  # Use self.window instead of creating new Tk()
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.resizable(False, False)
        
        # Create notebook for tabbed interface
        notebook = ttk.Notebook(settings_window)
        notebook.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Camera settings tab
        camera_frame = ttk.Frame(notebook)
        notebook.add(camera_frame, text='Camera')
        
        # Camera selection
        ttk.Label(camera_frame, text="Select Camera:").pack(pady=5)
        camera_combo = ttk.Combobox(camera_frame, values=["Camera 0", "Camera 1", "Camera 2"])
        camera_combo.pack(pady=5)
        camera_combo.set("Camera 0")

        # Resolution settings
        ttk.Label(camera_frame, text="Resolution:").pack(pady=5)
        resolution_combo = ttk.Combobox(camera_frame, values=["640x480", "1280x720", "1920x1080"])
        resolution_combo.pack(pady=5)
        resolution_combo.set(f"{self.window_width}x{self.window_height}")
        
        # Model settings tab
        model_frame = ttk.Frame(notebook)
        notebook.add(model_frame, text='Model')
        
        # Model selection
        ttk.Label(model_frame, text="Select Model:").pack(pady=5)
        model_combo = ttk.Combobox(model_frame, values=["best.pt", "yolov8n.pt", "yolov8s.pt", "best10epoch.pt"])
        model_combo.pack(pady=5)
        model_combo.set("best.pt")

        # Save button
        def save_settings():
            # Get resolution values
            width, height = map(int, resolution_combo.get().split('x'))
            self.window_width = width
            self.window_height = height
            
            # Update window size
            self.window.geometry(f"{width}x{height}")
            
            # Update camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            settings_window.destroy()
            
        save_button = ttk.Button(settings_window, text="Save", command=save_settings)
        save_button.pack(pady=10)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            if self.detection_active:
                # Perform detection with confidence threshold
                results = self.model(frame, conf=0.7)  # Set minimum confidence to 70%
                if len(results[0].boxes) > 0:  # Check if any detections above threshold
                    annotated_frame = results[0].plot()
                else:
                    annotated_frame = frame
            else:
                # Show raw frame without detection
                annotated_frame = frame
            
            # Get window dimensions with fallback values
            window_width = max(self.window.winfo_width(), 640)
            window_height = max(self.window.winfo_height(), 480)
            
            # Convert to format Tkinter can display
            cv_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv_image)
            
            # Resize image to fit window while maintaining aspect ratio
            aspect_ratio = pil_image.width / pil_image.height
            new_width = window_width
            new_height = window_height
            
            if window_width / window_height > aspect_ratio:
                new_width = int(window_height * aspect_ratio)
            else:
                new_height = int(window_width / aspect_ratio)
                
            # Only resize if dimensions are valid
            if new_width > 0 and new_height > 0:
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image=pil_image)
                self.video_label.config(image=self.photo)
            
        # Update frame every 10ms
        self.window.after(10, self.update_frame)

# Create and run app
root = Tk()
app = SafetyApp(root)
root.mainloop()