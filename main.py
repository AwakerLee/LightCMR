from DMKD import DMKD
from utils import logger
from args import config

def log_info(logger, config):

    logger.info('--- Configs List---')
    logger.info('--- Dadaset:{}'.format(config.DATASET))
    logger.info('--- Train:{}'.format(config.TRAIN))
    logger.info('--- Bit:{}'.format(config.HASH_BIT))
    logger.info('--- Eta:{}'.format(config.eta))
    logger.info('--- Beta:{}'.format(config.beta))
    logger.info('--- Lambda:{}'.format(config.lamb))
    logger.info('--- Mu:{}'.format(config.mu))
    logger.info('--- Batch:{}'.format(config.BATCH_SIZE))
    logger.info('--- Topk:{}'.format(config.topk))
    logger.info('--- Lr_IMG:{}'.format(config.LR_IMG))
    logger.info('--- Lr_TXT:{}'.format(config.LR_TXT))
    logger.info('--- L1:{}'.format(config.l1))
    logger.info('--- L2:{}'.format(config.l2))
    logger.info('--- L3:{}'.format(config.l3))
    logger.info('--- L4:{}'.format(config.l4))




def main():

        # log
        # if config.TRAIN == False:
        #     Model.load_checkpoints(config.CHECKPOINT)
        #     Model.eval()
        log = logger()
        # else:

        log_info(log, config)
        Model = DMKD(log, config)
        for epoch in range(config.NUM_EPOCH):
            Model.train(epoch)
            if (epoch + 1) % config.EVAL_INTERVAL == 0:
                Model.eval()

            #save the model
            if epoch + 1 == config.NUM_EPOCH:
                Model.save_checkpoints(config.CHECKPOINT)

if __name__ == '__main__':
    main()
