import yaml


def yaml_write(dic, file, safe=True):
    with open(file, 'w') as f:
        if safe:
            yaml.safe_dump(dic, f)
        else:
            yaml.dump(dic, f)
    return


def yaml_read(file):
    with open(file) as f:
        #         return yaml.load(f, Loader=yaml.FullLoader)
        return yaml.load(f, Loader=yaml.Loader)
