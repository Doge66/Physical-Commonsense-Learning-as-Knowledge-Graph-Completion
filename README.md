
## Learning Physical Common Sense as Knowledge Graph Completion via BERT Data Augmentation and Constrained Tucker Factorization

------

### Dependencies

* PyTorch 1.4.0

* OpenKE (Han et al., 2018)

### Triple Classification

* Ours in Table 1: python pc_triple_classification.py --add_constraint 

* Without data augmentation: python pc_triple_classification.py --data_dir ./data/kge/original/ --add_constraint

* Without constraint: python pc_triple_classification.py

### Link Prediction

* Ours in Table 2: python pc_link_prediction.py --add_constraint

* Without data augmentation: python pc_link_prediction.py --data_dir ./data/kge/original/ --add_constraint

* Without constraint: python pc_link_prediction.py

### OpenKE

The results of other methods in Table 1 and Table 2:

* TransE
    
    * python transe_eval.py

    * python openke_triple_classification.py --model transe

* TransD

    * python transd_eval.py

    * python openke_triple_classification.py --model transd

* RESCAL

    * python rescal_eval.py

    * python openke_triple_classification.py --model rescal

* DistMult

    * python distmult_eval.py

    * python openke_triple_classification.py --model distmult

* ComplEx

    * python complex_eval.py

    * python openke_triple_classification.py --model complex

* SimplE

    * python simple_eval.py

    * python openke_triple_classification.py --model simple

* Tucker

    * Triple classification: python tucker_triple_classification.py

    * Link prediction: python tucker_link_prediction.py

### Cross Validation

* Triple classification: python cv.py

* Link prediction: python cv_lp.py

### Acknowledgements

* The implementation is mainly modified from [Balazevic et al., 2019](https://github.com/ibalazevic/TuckER).

* The code of loading physical commonsense data and calculating evaluation metrics is taken from [Forbes et al., 2019](https://github.com/mbforbes/physical-commonsense).

### References

* Maxwell Forbes, Ari Holtzman, and Yejin Choi. 2019. Do neural language representations learn physical 457 commonsense? Proceedings of the 41st Annual 458 Conference of the Cognitive Science Society.

* Ivana Balazevic, Carl Allen, and Timothy Hospedales. 2019. TuckER: Tensor factorization for knowledge graph completion. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5185–5194, Hong Kong, China. Association for Computational Linguistics.

* Xu Han, Shulin Cao, Xin Lv, Yankai Lin, Zhiyuan Liu, Maosong Sun, and Juanzi Li. 2018. OpenKE: An open toolkit for knowledge embedding. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 139–144, Brussels, Belgium. Association for Computational Linguistics.


