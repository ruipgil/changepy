# changepy

Changepoint detection in time series in pure python

## Install

```
pip install changepy
```

## Examples

```python
    >>> from changepy import pelt
    >>> from changepy.costs import normal_mean
    >>> size = 100

    >>> mean_a = 0.0
    >>> mean_b = 10.0
    >>> var = 0.1

    >>> data_a = np.random.normal(mean_a, var, size)
    >>> data_b = np.random.normal(mean_b, var, size)
    >>> data = np.append(data_a, data_b)

    >>> pelt(normal_mean(data, var), len(data))
    [0, 100] # since data is random, sometimes it might be different, but most of the time there will be at most a couple more values around 100
```

For more examples see [pelt_test.py](./pelt_test.py)

## Reference

Currently there is only one algorithm for changepoint evaluation, the PELT algorithm [1].


The PELT algorithm requires a cost function. Currently there are three functions available through this library. However, you could implement your own, for your specific needs.
Those functions are:
  + ``` normal_mean ```, which expects normal distributed data, with changing mean
  + ``` normal_var ```, which expects normal distributed data, with changing variance
  + ``` normal_meanvar ```, which expects normal distributed data, with changing mean and variance
  + ``` poisson ```, which expect poisson distributed data, with changing mean
  + ``` exponential ```, which expect exponential distributed data, with changing mean


> Test with ``` python test_pelt.py ```

## Other implementations

This is mostly a port from other libraries, most of all from [STOR-i's changepoint package for julia](https://github.com/STOR-i/Changepoints.jl) and [rkillick cpt package for r](https://github.com/rkillick/changepoint/)


[1]: Killick R, Fearnhead P, Eckley IA (2012) Optimal detection of changepoints with a linear computational cost, JASA 107(500), 1590-1598

## License

[MIT](./LICENSE)

