from datetime import datetime


def timer(start_time=None):
    """
    Keep track of the time taken by a certain portion of code.

    Examples
    --------

    >>> start_time = timer(None)
    >>> # Run some code
    >>> timer(start_time)
    Time taken: 2 hours 25 minutes and 23 seconds.
    """
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
