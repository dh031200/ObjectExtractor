from utils import *


def main(_args):
    with Data(_args) as data:
        model = Model(data, _args.model, _args.verbose)
        for current, detect in enumerate(model.track()):
            if not _args.verbose:
                processing(current, data.total_frame)
            data.save(detect)
        data.parse_detect_results()
        data.save_results()


if __name__ == '__main__':
    args = parse_args()
    main(args)
