from params_and_parser import config
from algoritms import run_yolo
from compare_points_position import compare
las_path = config['las_path']
output_path = config["output"]
model_path = config["yolo_model_path"]

res = run_yolo(las_path,output_path,model_path)
compare(config["etalon_path"],output_path)


