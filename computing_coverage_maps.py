import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

# Set random seed for reproducibility
sionna.config.seed = 42

# importing other libraries
# Sionna imports
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera, watt_to_dbm
from sionna.mimo.precoding import normalize_precoding_power, grid_of_beams_dft
from sionna.utils import log10

# Python importings
import gc
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
try:
    import pandas as pd
except:
    os.system("pip install pandas")
    import pandas as pd

# file imports
#import custom_MSI


def extract_data(df, name):

    """
    This function retrieves coordinates, height, gains and tilt
    from the desired BS, given its ID
    """

    # print(df.head()) # Display the first few rows
    # print(df.dtypes)

    # preprocessing (preguntar a Rolando)        
    # filtered_df = df[(df['X(UTMHuso30)'] >= X_min) & (df['X(UTMHuso30)'] <= X_max) & (df['Y(UTMHuso30)'] >= Y_min) & (df['Y(UTMHuso30)'] <= Y_max)]
    # filtered_df.head()
    
    df_one_PROV_EMPLA = df[df["PROV_EMPLA"] == int(name)] # filter by PROV_EMPLA given by the name of the folder
    # print(f"Test whether the DF is empty or not: {df_one_PROV_EMPLA}")  # Display the filtered rows
    X_UTMHuso30 = df_one_PROV_EMPLA["X(UTMHuso30)"].tolist()
    Y_UTMHuso30 = df_one_PROV_EMPLA["Y(UTMHuso30)"].tolist()
    # print(f"Estas son las coordenadas: X_UTMHuso30: {X_UTMHuso30}")
    # print(f"Estas son las coordenadas: Y_UTMHuso30: {Y_UTMHuso30}")

    # let us convert the coordinates to an understandable format for Sionna --> relative location coordinates
    # x_location =  X_UTMHuso30 - X_min*np.ones_like(X_UTMHuso30) # ask Rolando
    # y_location =  Y_UTMHuso30 - Y_min*np.ones_like(Y_UTMHuso30)

    orientations = df_one_PROV_EMPLA["ORIENTACION"].tolist()
    gants = df_one_PROV_EMPLA["GANT"].tolist()
    alturas = df_one_PROV_EMPLA["altura"].tolist()
    tilts = df_one_PROV_EMPLA["tilt total"].tolist()

    return orientations, gants, alturas, tilts


# paramters definition
# X_min = 440000
# X_max = 450000
# Y_min = 4470000
# Y_max = 4480000
def main():
    cm_metric = "rss" # choose between: ["path_gain", "rss", "SINR"]
    delete_previous_results = False
    render_to_file = False

    # First, let's start by retrieving the coords of the BS by getting the name of the folder and searching by its name on the dataset
    directories = ["sionna_madrid" ,"sionna_madrid_concrete_ground"]
    for directory in directories:
        error_file_path = os.path.join(directory, "error_logs.txt")
        for name in os.listdir(directory):
            # if os.path.isdir(os.path.join(directory, name)) and name != ".ipynb_checkpoints" and name not in ["2800930", "2800754"]:
            if os.path.isdir(os.path.join(directory, name)) and name != ".ipynb_checkpoints":
                print(f"Element: {name}")
                # cwd = os.getcwd()
                scene_dir = os.path.join(directory, name)
                # print(scene_dir)

                # if desired, delete previous results to purge directories
                if delete_previous_results:
                    if os.path.isdir(os.path.join(directory, name)) and name != ".ipynb_checkpoints":
                        scene_dir = os.path.join(directory, name)
            
                        # let us delete the ancient PDFs
                        pdf_files = [file for file in os.listdir(scene_dir) if file.endswith(".pdf")]
                        for pdf in pdf_files:
                            path_pdf_i = os.path.join(scene_dir, str(pdf))
                            os.remove(path_pdf_i)
                        print("Finished deleting all ancient PDFs")

                        # now let us delete the ancient coverage maps
                        npy_files = [file for file in os.listdir(scene_dir) if file.endswith(".npy")]
                        for npy in npy_files:
                            path_npy_i = os.path.join(scene_dir, str(npy))
                            os.remove(path_npy_i)
                        print("Finished deleting all ancient npys")
            
                        # let us delete all the ancient rendered scenes
                        rendered_scene = [file for file in os.listdir(scene_dir) if file.endswith(".png")]
                        print(f"Rendered scenes: {rendered_scene}")
                        if rendered_scene:
                            os.remove(os.path.join(scene_dir, str(rendered_scene[0])))
                

                '''if pdf_files:
                    print("Skipping this DIR, CMs already computed!")
                    continue'''

                # else:

                # let us get the .xml file
                xml_file = [os.path.join(scene_dir, file) for file in os.listdir(scene_dir) if file.endswith(".xml")]
                print(f"XML file -> {xml_file}")

                # try to load the scene, if it fails save the log
                try: 
                    scene = sionna.rt.load_scene(xml_file[0])
                    scene_size_x = scene.size[0]
                    scene_size_y = scene.size[1]
                    print(f"We loaded the scene")
                    print(f"This is the size of the scene:{scene.size}")

                    # let us render the scene
                    my_cam = Camera("my_cam", position=[-250,250,150], look_at=[0,0,0])
                    scene.add(my_cam)
                    if render_to_file:
                        scene.render_to_file(camera="my_cam", filename=os.path.join(scene_dir,"rendered_scene.png"), resolution=[650,500])
                    scene.remove("my_cam")

                    # let us now specify the frequency of the scene
                    scene.frequency = 3.5e9
                    print(f"This is the frequency of the scene:{scene.frequency} ")

                except Exception as exception:
                    print(f"An error has occurred --> {exception}")
                    print(f"Checking the error_log file")
                    current_time = datetime.datetime.now() # to identify the date and time when the exception arised  

                    # Check if the file exists, if not, create it
                    if not os.path.exists(error_file_path):
                        with open(error_file_path, "w") as file:
                            file.write(f"{current_time} ---> Exception: {exception}.\n")  # Writing inside the file
                            print(f"File '{error_file_path}' created and error written successfully.")
                    else:
                        with open(error_file_path, "a") as file:  # Append mode to avoid overwriting
                            file.write(f"{current_time} ---> Exception: {exception}.\n")  # Writing inside the file
                            print(f"File '{error_file_path}' already exists. Added a new line.")

                    print(f"\nScene could not be loaded, saving the output on the .txt")
                    continue

                # the scene is loaded correctly, let us now retrieve the relevant antenna parameters
                df = pd.read_csv("datos_antenas_Madrid_191124.csv", sep=";") # Replace with your file path
                orientations, gants, alturas, tilts = extract_data(df,name)

                # let us compute the coverage map
                num_antennas = len(orientations)
                tx_power = 0 # If we define the Tx power as 0 dBm we are going to obtain instead of received signal the pathloss
                gain_tr38901 = 8 # double check the value in case of doubt

                # This loop defines each transmitter
                tensor_list = [] 
                for i in range(num_antennas):
                    print(f"Sector number {i}")
                    transmitter = Transmitter(name=f"tx_{i+1}",
                                            position=[0,0,alturas[i]],
                                            orientation = [np.deg2rad(orientations[i]), np.deg2rad(tilts[i]), 0], # ToDo adjust reference angles
                                            power_dbm = tx_power + gants[i] - gain_tr38901) # We are considering here the PIRE: ToDo Substract the gain of the radiation patter we use

                    scene.add(transmitter)
                    scene.tx_array = PlanarArray(num_rows=1,
                                                num_cols=1,
                                                vertical_spacing=0.5,  # relative to wavelength
                                                horizontal_spacing=0.5,  # relative to wavelength
                                                pattern="tr38901",
                                                polarization="V")

                    scene.rx_array = scene.tx_array

                    # define a camera to save the coverage map
                    camera_1 = Camera(name = 'testing_camera', position=[0, 0, 300], look_at=[0, 0, 0])

                    max_depth = 8
                    num_samples = int(10e6)
                    cm = scene.coverage_map(max_depth=max_depth,           # Maximum number of ray scene interactions
                                            num_samples=num_samples, # If you increase: less noise, but more memory required
                                            diffraction = True,
                                            cm_cell_size=(2, 2),   # Resolution of the coverage map
                                            cm_center=[0, 0, 1.5],   # Center of the coverage map, 1.5 m for the Rx_height
                                            cm_size=[int(scene_size_x), int(scene_size_y)],    # Total size of the coverage map
                                            cm_orientation=[0, 0, 0])
                    cm1 = cm
                    cm_image = cm.show(metric = cm_metric, vmin = -200, vmax = -50)

                    # if cm1 == cm: print(f"Both tensors are equal")

                    # let us save the CM
                    cm_image.savefig(os.path.join(scene_dir, f"sector_{orientations[i]}_md-{str(max_depth)}_ns-{str(num_samples)}.pdf"))

                    # let us save the CM as a tensor in both natural units and in dBs
                    cm_tensor_natural = cm.rss
                    # print(f"\n Original tensor with no conversion: {cm_tensor_natural}\n")

                    cm_tensor_dB = watt_to_dbm(cm_tensor_natural)
                    cm_tensor_dB = np.nan_to_num(cm_tensor_dB, nan=-220, posinf = -220, neginf = -220)

                    # cm_tensor_dB = 30 + 10*np.log10(cm_tensor_natural)
                    # cm_tensor_dB = np.nan_to_num(cm_tensor_dB, nan=-220, posinf = -220, neginf = -220)
                    # print(f"This is the tensor with np.log10: {cm_tensor_np_dB}")

                    # Avoid NaN values and + || - infty
                    min_val = tf.reduce_min(tf.where(tf.math.is_nan(cm_tensor_dB), tf.constant(float('inf')), cm_tensor_dB))
                    print(f"Min value of the tensor: {min_val}")
                    print(type(cm_tensor_dB))
                    print(tf.shape(cm_tensor_dB))

                    tensor_list.append(cm_tensor_dB)
                    print(f"This is the shape of the tensor with no mods:{tf.shape(cm_tensor_natural)}")
                    print(f"Content of the tensor_list: {tensor_list}")
                    
                    # has_nonzero = tf.reduce_any(cm_tensor_natural != 0)
                    # print(f"Elements inside the tensor != 0:{has_nonzero} ")
                    
                    plt.close("all") # close the image to avoid memory consumption

                    # Free memory properly
                    del cm, cm_tensor_natural, cm_tensor_dB
                    gc.collect()  # Force memory cleanup

                    scene.remove(f"tx_{i+1}")

                final_tensor = tf.stack(tensor_list, axis = 0)
                print(f"Shape of the final tensor: {tf.shape(final_tensor)}")
                print(f"Content of the final tensor: {final_tensor}")

                # let's turn it into a npy variable
                # final_tensor_npy = final_tensor.numpy()

                # let's save the tensor in a .npy file
                np.save(os.path.join(scene_dir,"cm_tensor_dB.npy"), final_tensor.numpy().astype(np.float32))

                # clean all tensor variables
                tensor_list = [] 
                final_tensor = []
                tf.keras.backend.clear_session()
                
                # Clear scene after each folder
                del scene, df
                gc.collect()

main()
# scene.preview()
