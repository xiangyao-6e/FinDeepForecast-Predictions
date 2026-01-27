import logging


def get_logger():
    """Return the shared pipeline logger (use after setup_logger has been called from run_all_with_extraction)."""
    return logging.getLogger('pipeline')


def setup_logger(output_dir_path=None, log_level=logging.INFO):
    """
    Set up the pipeline logger. When output_dir_path is provided (e.g. from run_all_with_extraction),
    log to pipeline.log in that directory. Otherwise add a console handler for standalone use.
    """
    logger = logging.getLogger('pipeline')

    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if output_dir_path is not None:
        log_file_path = output_dir_path / "pipeline.log"
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging initialized. Log file: {log_file_path}")
    else:
        # console_handler = logging.StreamHandler()
        # console_handler.setFormatter(formatter)
        # logger.addHandler(console_handler)
        raise Exception("Output directory path is required to setup logger")

    return logger

def update_logger(log_file='app.log'):
    logger = logging.getLogger(__name__)
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            h.close()
    # File handler
    # Formatter (how logs will look)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def pretty_log_research_result(
    logger: logging.Logger, final_output: str, web_searches: list, reasoning_steps: list, citations: list
):
    logger.info('📄 Deep Research Results:')
    logger.info('=' * 50)
    logger.info(final_output)

    if web_searches:
        logger.info(f'🔍 Web Searches Performed ({len(web_searches)}):')
        for i, search in enumerate(web_searches, 1):
            logger.info(f'{i}. {search}')

    if reasoning_steps:
        logger.info(f'🧠 Reasoning Steps ({len(reasoning_steps)}):')
        for i, step in enumerate(reasoning_steps, 1):
            logger.info(f'{i}. {step}')

    if citations:
        logger.info(f'📚 Citations ({len(citations)}):')
        for i, citation in enumerate(citations, 1):
            logger.info(f'{i}. {citation.title}')
            logger.info(f'   URL: {citation.url}')
