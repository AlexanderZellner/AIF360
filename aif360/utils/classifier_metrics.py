# Metrics function
from collections import OrderedDict
from aif360.metrics import ClassificationMetric


class ClassifierMetricUtils:
    def __init__(self,dataset_true, dataset_pred,
                unprivileged_groups, privileged_groups):
        metric: ClassificationMetric

        self.metric = ClassificationMetric(dataset_true,
                                           dataset_pred,
                                           unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)

    def get_metrics(self):
        metrics = OrderedDict()
        metrics["Balanced accuracy"] = 0.5 * (self.metric.true_positive_rate() +
                                              self.metric.true_negative_rate())
        metrics["Statistical parity difference"] = self.metric.statistical_parity_difference()
        metrics["Disparate impact"] = self.metric.disparate_impact()
        metrics["Average odds difference"] = self.metric.average_odds_difference()
        metrics["Equal opportunity difference"] = self.metric.equal_opportunity_difference()
        metrics["Theil index"] = self.metric.theil_index()

    def get_explanation(self):
        explanations = OrderedDict()
        explanations["Balanced accuracy"] = 0.5 * (self.metric.true_positive_rate() +
                                                   self.metric.true_negative_rate())
        explanations["Statistical parity difference"] = self.metric.statistical_parity_difference()
        explanations["Disparate impact"] = self.metric.disparate_impact()
        explanations["Average odds difference"] = self.metric.average_odds_difference()
        explanations["Equal opportunity difference"] = self.metric.equal_opportunity_difference()
        explanations["Theil index"] = self.metric.theil_index()

    def compute_metrics(self):
        metrics = OrderedDict()
        metrics["Balanced accuracy"] = 0.5 * (self.metric.true_positive_rate() +
                                              self.metric.true_negative_rate())
        metrics["Statistical parity difference"] = self.metric.statistical_parity_difference()
        metrics["Disparate impact"] = self.metric.disparate_impact()
        metrics["Average odds difference"] = self.metric.average_odds_difference()
        metrics["Equal opportunity difference"] = self.metric.equal_opportunity_difference()
        metrics["Theil index"] = self.metric.theil_index()

        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))

    def explain_metrics(self):
        explainer_transf_train = MetricTextExplainer(self.metric)

        explanations = OrderedDict()
        explanations["Balanced accuracy"] = explainer_transf_train.accuracy()
        explanations["Statistical parity difference"] = explainer_transf_train.statistical_parity_difference()
        explanations["Disparate impact"] = explainer_transf_train.disparate_impact()
        explanations["Average odds difference"] = explainer_transf_train.average_odds_difference()
        explanations["Equal opportunity difference"] = explainer_transf_train.equal_opportunity_difference()
        explanations["Theil index"] = explainer_transf_train.theil_index()

        for exp in explanations:
            print(explanations[exp])



