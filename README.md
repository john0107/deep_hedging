# A Deep Learning Approach to Hedging Financial Derivatives with Price Impact

This repository contains an implementation of a deep learning framework for hedging financial derivatives in markets with frictions which is the topic of my bachelor's thesis.

The basic idea is to represent hedging strategies by deep neural networks and to utilize the efficient optimization algorithms known for neural networks to obtain an optimal hedging strategy.
This approach is not dependent on specific market dynamics and is thus entirely data-driven.
In particular, it allows modelling market frictions such as transaction costs, liquidity constraints and market impact.
This approach was first proposed in [[Büh+19]][deep-hedging-wp].

The goal of my thesis is to investigate the performance of the deep hedging approach in the presence of price impact.
For this purpose, we consider a financial model with permanent price impact, as studied in [[BLZ16]][bouchard], and derive a perfect hedging strategy in a special case.
Our numerical experiments show promising results for the studied impact structure.


## References

[[Büh+19]][deep-hedging-wp] Hans Bühler, Lukas Gonon, Josef Teichmann and Ben Wood, "[Deep Hedging][deep-hedging-wp]". In: Quantitative Finance 19.8 (2019),
pp. 1271–1291.

[[BLZ16]][bouchard] Bruno Bouchard, Grégoire Loeper, and Yiyi Zou. "[Almost-sure hedging with permanent price impact][bouchard]". In: Finance and Stochastics 20.3 (2016), pp. 741–771.

[deep-hedging-wp]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3355706
[bouchard]: https://link.springer.com/article/10.1007/s00780-016-0295-1
