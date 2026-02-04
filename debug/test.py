#!/usr/bin/env python3
import argparse
import glob
import os
import time

from algoritms import run_yolo
from params_and_parser import config
#from algoritms import cirle_v,yolo_v
def main():
    start = time.time()
    parametrs = config
    print(parametrs)
    yolo_model_path = parametrs['yolo_model_path'],

    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Входной LAS')
    parser.add_argument('--model', required=True, help='путь до модели ')
    parser.add_argument('--output', required=True, help='Куда сохранить итоговый CSV')
    args = parser.parse_args()

    las_files = glob.glob(os.path.join(args.input, '*NORM.las'))

    #print(f'СОДЕРЖИМОЕ РАБОЧЕЙ ДИРЕКТОРИИ:{os.listdir(args.input)}')
    print(f'las файлы в рабочей директории:{las_files}')

    run_yolo(las_files[0], args.output, args.model)

    #для проверки на локалке после прогона
    #from main import compare_result,read_data,match_xyhs_pairs
    #shp_path = 'F:/pp57/etalon/pp57-res.shp'
    #csv_path = args.output
    #compare_result(shp_path, csv_path,radius_threshold=0.04,height_threshold=2.23)


    end = time.time()
    print(f'total time: {end - start}')

if __name__ == '__main__':
    main()
