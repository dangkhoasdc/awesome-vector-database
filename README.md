# :mag: Awesome Vector Database [![Awesome](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of awesome works related to high dimensional structure/vector search &amp; database 

# Services

- [Pinecone](https://www.pinecone.io/).
- [Weaviate](https://github.com/weaviate/weaviate).
- [Vespa](https://vespa.ai/)
- [txtai](https://github.com/neuml/txtai)


# Libraries & Engines
## Multidimensional data / Vectors

- :star: [Faiss](https://faiss.ai/).
- [Typesense](https://typesense.org/).
- [Qdrant](https://qdrant.tech/).
- [annoy](https://github.com/spotify/annoy)
- [NGT](https://github.com/yahoojapan/NGT)

## Texts
- [OpenSearch](https://opensearch.org/).
- [PISA](https://github.com/pisa-engine/pisa)
- [Tantivy](https://github.com/quickwit-oss/tantivy)
- [sonic](https://github.com/valeriansaliou/sonic)

# Benchmarks & Databases

- [ANN Benchmarks](http://ann-benchmarks.com/) [[Paper](https://arxiv.org/pdf/1807.05614.pdf)].
- [Billion-scale ANNS Benchmarks](https://big-ann-benchmarks.com).
- [BEIR](https://github.com/beir-cellar/beir)

# Books
- [Foundations of Multidimensional and Metric Data Structures](https://www.amazon.com/Foundations-Multidimensional-Structures-Kaufmann-Computer/dp/0123694469/)
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/information-retrieval-book.html)

# Conferences & Workships
- :star: [VLDB](https://vldb.org)
- :star: [Image Retrieval in the Wild (CVPR20)](https://matsui528.github.io/cvpr2020_tutorial_retrieval/) [[Video](https://www.youtube.com/watch?v=SKrHs03i08Q)]
- [Neural Search In Action](https://matsui528.github.io/cvpr2023_tutorial_neural_search/)
- ACM MM 2020: [Effective and Efficient: Toward Open-world Instance Re-identification](https://wangzwhu.github.io/home/acmmm2020_tutorial_reid.html)
  - Billion-scale Approximate Nearest Neighbor Search: [[Slides](https://wangzwhu.github.io/home/file/acmmm-t-part3-ann.pdf), [Video](https://www.youtube.com/watch?v=iI8e3kU11eU)].
  - Is instance search a solved problem? [[Slides](https://wangzwhu.github.io/home/file/acmmm-t-part4-ins.pdf), [Video](https://www.youtube.com/watch?v=cH256Zqt5Ms)].

# Publications
## Quantization
![](https://raw.githubusercontent.com/wiki/facebookresearch/faiss/PQ_variants_Faiss_annotated.png)
Source: A survey of product quantization.

- :star: PQ: Product quantization for nearest neighbor search. Jegou, Herve, Matthijs Douze, and Cordelia Schmid. [[Paper](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf), [Code](https://github.com/facebookresearch/faiss), [Julia Code](https://github.com/una-dinosauria/Rayuela.jl), [nanopq](https://github.com/matsui528/nanopq)]
- :star: k-selection on GPU: Billion-scale similarity search with gpus. Johnson, Jeff, Matthijs Douze, and Hervé Jégou [[Paper](https://arxiv.org/pdf/1702.08734.pdf), [Code](https://github.com/facebookresearch/faiss)]
- :star: A survey of product quantization. Matsui, Yusuke, Yusuke Uchida, Hervé Jégou, and Shin'ichi Satoh [[Paper](https://www.jstage.jst.go.jp/article/mta/6/1/6_2/_pdf)]
- OPQ: Optimized Product Quantization. Ge, Tiezheng, Kaiming He, Qifa Ke, and Jian Sun [[Homepage](https://kaiminghe.github.io/cvpr13/index.html), [Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf), [Code](https://kaiminghe.github.io/cvpr13/matlab_OPQ_release_v1.1.rar), [nanopq](https://github.com/matsui528/nanopq)]
- Quicker adc: Unlocking the hidden potential of product quantization with simd. André, Fabien, Anne-Marie Kermarrec, and Nicolas Le Scouarnec [[Paper](https://arxiv.org/pdf/1812.09162), [Code](https://github.com/technicolor-research/faiss-quickeradc)]
  - Accelerated nearest neighbor search with quick adc. André, Fabien, Anne-Marie Kermarrec, and Nicolas Le Scouarnec [[Paper](https://arxiv.org/pdf/1704.07355.pdf)].
  - Cache locality is not enough: High-performance nearest neighbor search with product quantization fast scan. Fabien André, Anne-Marie Kermarrec, Nicolas Le Scouarnec [[Paper](https://hal.inria.fr/hal-01239055/document)]
- ScaNN: Accelerating Large-Scale Inference with Anisotropic Vector Quantization. Guo, Ruiqi, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar [[Paper](http://proceedings.mlr.press/v119/guo20h/guo20h.pdf), [Python/C++ Inference](https://github.com/google-research/google-research/tree/master/scann), [Julia Training/Inference](https://github.com/AxelvL/AHPQ.jl)]
- The inverted multi-index. Babenko, Artem, and Victor Lempitsky [[Paper](https://cmp.felk.cvut.cz/~toliageo/rg/papers/BabenkoLempitsky_PAMI2014_The%20Inverted%20Multi-Index.pdf), [Code](https://github.com/jatin7gupta/Product-Quantization)]

## Graph-based Methods
- :star: HNSW: Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. Malkov, Yu A., and Dmitry A. Yashunin. [[Paper](https://arxiv.org/pdf/1603.09320.pdf), [Code](https://github.com/nmslib/hnswlib)]

## Hashing
- :star: [Awesome Papers on Learning to Hash](https://learning2hash.github.io)
- :star: A survey on learning to hash. Wang, Jingdong, Ting Zhang, Nicu Sebe, and Heng Tao Shen [[Paper](https://arxiv.org/pdf/1606.00185.pdf)].
- :star: A survey on deep hashing methods. Luo, Xiao, Haixin Wang, Daqing Wu, Chong Chen, Minghua Deng, Jianqiang Huang, and Xian-Sheng Hua. [[Paper](https://dl.acm.org/doi/full/10.1145/3532624)]
- :star: Iterative quantization: A procrustean approach to learning binary codes for large-scale image retrieval. Gong, Yunchao, Svetlana Lazebnik, Albert Gordo, and Florent Perronnin [[Paper](https://slazebni.cs.illinois.edu/publications/ITQ.pdf), [Python code](https://github.com/twistedcubic/learn-to-hash/blob/master/itq.py), [Matlab code](https://github.com/dangkhoasdc/sah/tree/master/itq)]


