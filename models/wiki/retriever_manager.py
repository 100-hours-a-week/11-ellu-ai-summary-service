import logging
from typing import Dict, Type
from app.config import RETRIEVER_CLASS

logger = logging.getLogger(__name__)

class RetrieverManager:    
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, retriever_class: Type):
        cls._registry[name] = retriever_class
        logger.info(f"Registered retriever: {name}")
    
    @classmethod
    def create_retriever(cls, class_name: str = None):
        logger.info(f"create_retriever called with class_name: {class_name} (type: {type(class_name)})")
            
        if class_name is None:
            class_name = RETRIEVER_CLASS
            logger.info(f"Using default retriever class: {class_name}")

        if class_name not in cls._registry:
            logger.warning(f"Unknown retriever class: {class_name}, falling back to BasicRetriever")
            class_name = "BasicRetriever"
        
        retriever_class = cls._registry[class_name]
        logger.info(f"Creating retriever: {class_name}")
        return retriever_class()
    
    @classmethod
    def get_available_retrievers(cls):
        return list(cls._registry.keys())

def register_all_retrievers():
    from models.wiki.retriever.basic_retriever import BasicRetriever
    # from models.wiki.retriever.query_enhanced_retriever import QueryEnhancedRetriever  
    # from models.wiki.retriever.role_enhanced_retriever import RoleEnhancedRetriever
    # from models.wiki.retriever.psuedo_expert_retriever import PsuedoExpertRetriever
    
    RetrieverManager.register("BasicRetriever", BasicRetriever)
    # RetrieverManager.register("QueryEnhancedRetriever", QueryEnhancedRetriever)
    # RetrieverManager.register("RoleEnhancedRetriever", RoleEnhancedRetriever)
    # RetrieverManager.register("PsuedoExpertRetriever", PsuedoExpertRetriever)

register_all_retrievers()