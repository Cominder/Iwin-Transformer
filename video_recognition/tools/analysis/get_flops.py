import argparse

from mmcv import Config

from mmaction.models import build_recognizer

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[340, 256],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    # if len(args.shape) == 1:
    #     input_shape = (1, 3, args.shape[0], args.shape[0])
    # elif len(args.shape) == 2:
    #     input_shape = (
    #         1,
    #         3,
    #     ) + tuple(args.shape)
    # elif len(args.shape) == 4:
    #     # n, c, h, w = args.shape
    #     input_shape = tuple(args.shape)
    # elif len(args.shape) == 5:
    #     # n, c, t, h, w = args.shape
    #     input_shape = tuple(args.shape)
    # else:
    #     raise ValueError('invalid input shape')
    

    # 修改输入形状，添加时间维度 D
    if len(args.shape) == 1:
        input_shape = (1, 3, 8, args.shape[0], args.shape[0])  # 默认 D=8
    elif len(args.shape) == 2:
        input_shape = (1, 3, 8) + tuple(args.shape)  # 默认 D=8
    elif len(args.shape) == 4:
        input_shape = tuple(args.shape)  # 用户指定完整形状
    elif len(args.shape) == 5:
        input_shape = tuple(args.shape)  # 用户指定完整形状
    else:
        raise ValueError('invalid input shape')

    # ... 其他代码保持不变 ...

    cfg = Config.fromfile(args.config)
    model = build_recognizer(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    model = model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
