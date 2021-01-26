import seaborn as sns
import yaml


class GlobalConfig:
    class __Config:
        def __init__(self):
            """
            Define all config variables. Will be accessible with normal autocomplete suggestions
            """
            self.edge_detection_type = None
            self.image_size = None
            self.visualization_dict = None

            self.clustering_eps = None
            self.clustering_min_sample = None
            self.mask_size = None
            self.n_random_points = None
            self.kernel_harris_corner_detector = None

            self.path_output = None
            self.path_summary = None
            self.dir_results = None
            self.dir_result_details = None
            self.file_name_input_image = None
            self.file_name_plane_masks = None
            self.file_name_planercnn_image = None

            '''
            Other (not loaded from config file)
            '''
            # Color pallet for visualization of masks
            self.color_pallet = [tuple([int(color[2] * 255), int(color[1] * 255), int(color[0] * 255)]) for color in
                                 sns.color_palette("bright", 10)]

    __instance: "__Config" = None

    @classmethod
    def get_config(cls) -> "__Config":
        """
        Access config as singleton, i.e. keep only one global config
        :return: global config instance
        """
        if cls.__instance is None:
            cls.__instance = cls.__Config()
        return cls.__instance

    @classmethod
    def load_config(cls, path) -> "__Config":
        """
        Load config from yaml and save it in the config instance for global access
        :param path: path to yaml config file
        :return: global config instance
        """
        unused_input = []
        cfg = cls.get_config()
        with open(path, 'r') as stream:
            input_config = yaml.safe_load(stream)
            for key, val in input_config.items():
                if key in cfg.__dict__.keys():  # save all keys from yaml that have the same name as __Config variables
                    setattr(cfg, key, val)
                else:
                    unused_input.append(key)
        print(f"WARNING: "
              f"The following keys specified in input file were not automatically saved in the config: {unused_input}")

        cfg = cls.__set_special_config_values(cfg, input_config)
        cls.__check_all_config_values_set(cfg)
        cls.__instance = cfg
        return cls.__instance

    @staticmethod
    def __set_special_config_values(cfg: __Config, config: dict) -> "__Config":
        """
        Handle special conversions from yaml input to a needed formats (i.e. splitting of lists)
        :param cfg: global config instance
        :param config: config dictionary from yaml
        :return: updated global config instance
        """
        cfg.file_name_plane_masks = lambda i: str(i) + config['file_name_plane_mask_suf']
        cfg.file_name_planercnn_image = lambda i: str(i) + config['file_name_planercnn_image_suf']
        cfg.dir_results = f"{cfg.edge_detection_type}"  # will be the output folder, create in data dir
        cfg.image_size = tuple(int(x) for x in config['image_size'].split(" "))
        return cfg

    @staticmethod
    def __check_all_config_values_set(cfg: __Config):
        """
        Check if all values of config class have been set and warn about unset values
        :param cfg: global config instance
        """
        unset_values = [key for key, val in cfg.__dict__.items() if val is None]
        if len(unset_values) > 0:
            print(f"WARNING: The following config variables have not been set: {unset_values}")


logging_config = {
    'format': '%(asctime)s [%(levelname)s]: %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S',
    'level': 20,  # info (see https://docs.python.org/3/library/logging.html#logging-levels)
}
