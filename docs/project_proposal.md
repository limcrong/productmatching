Background/Aims/Objectives:
E-commerce marketplaces frequently contain multiple listings representing the same underlying product, but with different titles, images and descriptions. The aim is to develop a pattern recognition system that identifies equivalent product listings using textual and visual information, enabling consumers to compare offers across merchants


Project Descriptions:
This project aims to develop a multimodal product matching system that can identify equivalent product listings across ecommerce merchants despite variations in product titles, descriptions and images. The system will enable multiple listings of the same underlying product to be grouped together, providing the foundation for consumers to compare available merchant offers and prices.

The primary dataset will be the Shopee Product Matching dataset, which contains ecommerce product listings with product titles, product images and labelled product groups indicating listings representing the same underlying product. These ground truth groupings will allow different product matching approaches to be trained and quantitatively evaluated. https://www.kaggle.com/competitions/shopee-product-matching/data

The proposed system will consist of the following main components:

1.	Text based product matching – Product titles and other available textual attributes will be processed to identify semantically similar product listings using machine learning and deep learning techniques.
2.	Image based product matching – Visual features extracted from product images will be used to identify visually similar products and provide an additional matching signal independent of textual descriptions.
3.	Multimodal product matching – Textual and visual representations will be combined using a hybrid or ensemble approach to determine whether different listings correspond to the same underlying product.
4.	Product retrieval and grouping – Candidate products will be retrieved based on similarity and grouped according to the predicted product identity, allowing equivalent listings from different sellers or merchants to be compared.
5.	Product matching prototype – A runnable prototype will demonstrate the application of the developed product matching models using product listings from the dataset. Given a product listing or reference product, the system will retrieve candidate listings, identify predicted matches and present the matching results together with their respective confidence or similarity scores. As a potential extension, the prototype may be enhanced into a consumer-facing product comparison application incorporating product listings and offer information from external data sources.

The different approaches will be evaluated quantitatively against labelled ground-truth product matches using suitable metrics such as precision, recall and F1 score. Experiments will compare text only, image only and multimodal approaches to determine whether combining multiple information sources improves product matching performance. Additional analysis will examine challenging cases such as visually similar product variants, accessories and different product generations.

The final deliverable will be a runnable pattern recognition system that demonstrates and evaluates the use of multimodal machine learning for product identification and matching. Where feasible, the system may be further extended to demonstrate how product matching can support cross-merchant product discovery and comparison in a consumer-facing use case.


