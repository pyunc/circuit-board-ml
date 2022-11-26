from diagrams import Cluster, Diagram
from diagrams.aws.compute import Lambda
from diagrams.aws.ml import Sagemaker
from diagrams.aws.storage import S3
from diagrams.onprem.analytics import Databricks
from diagrams.onprem.queue import Kafka
from diagrams.programming.language import Go

with Diagram("circuit-board-ml", show=True):

    datalake = S3("Datalake")

    pcg_databricks = Databricks("Personalized\nCandidate Generation")
    npcg_databricks = Databricks("Non Personalized\nCandidate Generation")
    feature_processor = Databricks("Feature Processor")

    p_recplat_folder = S3("Personalized Candidates")
    np_recplat_folder = S3("Non Personalized\nCandidates")
    batch_features_folder = S3("Batch Features")

    with Cluster("Batch Transform\nPersonalized"):
        sgmk_p_group = [Sagemaker("Candunga"), Sagemaker("Oiaso")]

    with Cluster("Batch Transform\nNon Personalized"):
        sgmk_np_group = [
            Sagemaker("Candunga"),
        ]

    with Cluster("Coookbook Model"):
        databricks_train_jobs = [
            Databricks("Train\nDataset"),
            Databricks("Context Feature\nDataset"),
        ]
        sgmk_cookbook_train = Sagemaker("Cookbook Train")
        sgmk_cookbook_serve = Sagemaker("Cookbook Endpoint")
        # sgmk_cookbook_train >> sgmk_cookbook_serve

    with Cluster("Feature Store"):
        feature_store = Go("Feature Store")

    with Cluster("Recplat"):
        feature_publisher = Lambda("Feature Publisher")
        kafka = Kafka("Kafka")
        feature_consumer = Go("Feature Consumer")
        recplat = Go("Recplat")

        feature_publisher >> kafka >> feature_consumer >> recplat

    pcg_databricks >> p_recplat_folder >> sgmk_p_group >> batch_features_folder
    npcg_databricks >> np_recplat_folder >> sgmk_np_group >> batch_features_folder

    batch_features_folder >> feature_publisher
    batch_features_folder >> feature_processor >> datalake

    databricks_train_jobs >> sgmk_cookbook_train >> sgmk_cookbook_serve
    datalake >> databricks_train_jobs
    recplat >> sgmk_cookbook_serve
    feature_store >> sgmk_cookbook_serve
