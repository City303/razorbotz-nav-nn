import torch
import yolov3, os

def call_command(param_list, command, keyargs):
    params = {}
    for kw in keyargs:
        if kw in param_list:
            params[kw] = keyargs[kw]

    for pm in param_list:
        if pm not in params.keys():
            params[pm] = None

    print('\nParameters:')
    for k, v in params.items():
        print(f'    {k:16} : {v}')
    
    for k, v in params.items():
        if v != None:
            command = command + ' --' + str(k) + ' ' + str(v)

    print('\nRunning command:')
    print('   ', command)
    os.system(command)


def run_detect(**kwargs):
    """Sets up the parameters and launch args for detect.py,
    and then calls the script.
    """
    cmd = 'python yolov3/detect.py'
    pms_list = [
        'image_folder', 'model_def', 
        'weights_path', 'class_path', 
        'conf_thres', 'nms_thres',
        'batch_size', 'n_cpu', 
        'img_size', 'checkpoint_model'
    ]
    call_command(pms_list, cmd, kwargs)



def run_test(**kwargs):
    """Sets up the parameters and launch args for test.py,
    and then calls the script.
    """
    cmd = 'python yolov3/test.py'
    pms_list = [
        'batch_size', 'model_def',
        'data_config', 'weights_path',
        'class_path', 'iou_thres',
        'nms_thres', 'conf_thres',
        'n_cpu', 'img_size'
    ]
    call_command(pms_list, cmd, kwargs)



def run_train(**kwargs):
    """Sets up the parameters and launch args for train.py,
    and then calls the script.
    """
    cmd = 'python yolov3/train.py'
    pms_list = [
        'epochs', 'batch_size',
        'gradient_accumulations', 'model_def',
        'data_config', 'pretrained_weights',
        'n_cpu', 'img_size', 'checkpoint_interval',
        'evaluation_interval', 'compute_map',
        'multiscale_training', 'verbose',
        'logdir'
    ]
    call_command(pms_list, cmd, kwargs)


if __name__ == '__main__':
    print('CUDA available:', torch.cuda.is_available())
    run_detect(image_folder='/dev/null')
    #run_test()
    #run_train()