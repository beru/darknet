#include "darknet.h"
#include "xplat.h"
#include <assert.h>

void train_segmenter(char *datacfg,
                     char *cfgfile,
                     char *weightfile,
#ifdef GPU
                     int *gpus,
                     int ngpus,
#endif
                     int clear)
{
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    srand(time(0));
#ifdef GPU
    printf("%d\n", ngpus);
    network *nets = xplat_malloc(ngpus, sizeof(network));
    int seed = rand();
    for (int i = 0; i < ngpus; ++i) {
        srand(seed);
        cuda_set_device(gpus[i]);
        parse_network_cfg(&nets[i], cfgfile);
        if (weightfile) {
            load_weights(&nets[i], weightfile);
        }
        if (clear) *nets[i].seen = 0;
        nets[i].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];
#else
    network net = {0};
    parse_network_cfg(&net, cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    if (clear) *net.seen = 0;
#endif

    int imgs = net.batch * net.subdivisions
#ifdef GPU
               * ngpus
#endif
    ;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n",
           net.learning_rate, net.momentum, net.decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *train_list = option_find_str(options, "train", "data/train.list");

    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    clock_t time;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
#ifdef THREAD
    args.threads = 32;
#endif
    args.min = net.min_crop;
    args.max = net.max_crop;
    args.angle = net.angle;
    args.aspect = net.aspect;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.size = net.w;
    args.classes = 80;

    args.paths = paths;
    args.n = imgs;
    args.m = N;
    args.type = SEGMENTATION_DATA;

    data train;
    data buffer;
    args.d = &buffer;
#ifdef THREAD
    pthread_t load_thread = load_data(args);
#endif
    int epoch = (*net.seen) / N;
    while (get_current_batch(&net) < net.max_batches || net.max_batches == 0) {
        time = clock();
#ifdef THREAD
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);
#else
        load_data(args);
        train = buffer;
#endif
        printf("Loaded: %lf seconds\n", sec(clock() - time));
        time=clock();

        float loss = 0;
#ifdef GPU
        if (ngpus == 1) {
            loss = train_network(&net, train);
        }else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(&net, train);
#endif
        if (avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n",
               get_current_batch(&net), (float)(*net.seen) / N, loss, avg_loss,
               get_current_rate(&net), sec(clock() - time), *net.seen);
        free_data(train);
        if (*net.seen / N > epoch) {
            epoch = *net.seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(&net, buff);
        }
        if (get_current_batch(&net)%100 == 0) {
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(&net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(&net, buff);

    free_network(&net);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    xplat_free(base);
}

void predict_segmenter(char *datafile, char *cfgfile, char *weightfile, char *filename)
{
    network net = {0};
    parse_network_cfg(&net, cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);

    clock_t time;
    char buff[256];
    char *input = buff;
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
        }else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image sized = letterbox_image(im, net.w, net.h);

        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(&net, X);
        image m = float_to_image(sized.w, sized.h, 81, predictions);
        image rgb = mask_to_rgb(m);
        show_image(sized, "orig");
        show_image(rgb, "pred");
#ifdef OPENCV
        cvWaitKey(0);
#endif
        printf("Predicted: %f\n", predictions[0]);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));
        free_image(&im);
        free_image(&sized);
        free_image(&rgb);
        if (filename) break;
    }
}


void demo_segmenter(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    printf("Regressor Demo\n");
    network net = {0};
    parse_network_cfg(&net, cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    srand(2222222);
    CvCapture * cap;

    if (filename) {
        cap = cvCaptureFromFile(filename);
    }else {
        cap = cvCaptureFromCAM(cam_index);
    }

    if (!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow("Regressor", CV_WINDOW_NORMAL); 
    cvResizeWindow("Regressor", 512, 512);
    float fps = 0;

    while (1) {
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        image in_s = letterbox_image(in, net.w, net.h);
        show_image(in, "Regressor");

        float *predictions = network_predict(&net, in_s.data);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        printf("People: %f\n", predictions[0]);

        free_image(&in_s);
        free_image(&in);

        cvWaitKey(10);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}


void run_segmenter(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

#ifdef GPU
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if (gpu_list) {
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        for (int i = 0; i < len; ++i) {
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = xplat_malloc(ngpus, sizeof(int));
        for (int i = 0; i < ngpus; ++i) {
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    }else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }
#endif

    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int clear = find_arg(argc, argv, "-clear");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if (0 == strcmp(argv[2], "test")) predict_segmenter(data, cfg, weights, filename);
    else if (0 == strcmp(argv[2], "train")) {
        train_segmenter(data,
                        cfg,
                        weights,
#ifdef GPU
                        gpus, ngpus,
#endif
                        clear);
    }
    else if (0 == strcmp(argv[2], "demo")) demo_segmenter(data, cfg, weights, cam_index, filename);
}


