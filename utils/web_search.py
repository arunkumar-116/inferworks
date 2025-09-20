# from tavily import TavilyClient
# from config.config import config
# import logging
# from typing import List, Dict, Any

# class WebSearchTool:
#     def __init__(self):
#         self.client = TavilyClient(api_key=config.TAVILY_API_KEY)
    
#     def search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
#         """Perform web search using Tavily with medical research focus"""
#         try:
#             # Add medical research context to the query
#             enhanced_query = f"medical research {query}" if self._is_medical_query(query) else query
            
#             response = self.client.search(
#                 query=enhanced_query,
#                 search_depth="advanced",
#                 max_results=max_results,
#                 include_domains=["nih.gov", "who.int", "thelancet.com", "nejm.org", "jamanetwork.com"]
#             )
#             return response
#         except Exception as e:
#             logging.error(f"Error performing web search: {e}")
#             return {"results": []}
    
#     def format_search_results(self, search_results: Dict[str, Any]) -> str:
#         """Format search results for context with medical research focus"""
#         if not search_results.get("results"):
#             return "No relevant medical research found in web search."
        
#         formatted_results = "Recent medical research from web search:\n\n"
#         for i, result in enumerate(search_results["results"], 1):
#             formatted_results += f"Research {i}:\n"
#             formatted_results += f"Title: {result.get('title', 'N/A')}\n"
#             formatted_results += f"Content: {result.get('content', 'N/A')}\n"
#             formatted_results += f"URL: {result.get('url', 'N/A')}\n\n"
        
#         return formatted_results
    
#     def _is_medical_query(self, query: str) -> bool:
#         """Determine if the query is medical in nature"""
#         medical_terms = [
#             'disease', 'treatment', 'patient', 'medical', 'health', 
#             'medicine', 'clinical', 'symptom', 'diagnosis', 'therapy',
#             'drug', 'pharmaceutical', 'hospital', 'doctor', 'nurse'
#         ]
        
#         query_lower = query.lower()
#         return any(term in query_lower for term in medical_terms)


from tavily import TavilyClient
from config.config import config
import logging
from typing import List, Dict, Any

class WebSearchTool:
    def __init__(self):
        self.client = TavilyClient(api_key=config.TAVILY_API_KEY)
    
    def search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Perform web search using Tavily with medical research focus"""
        try:
            # Add medical research context to the query
            enhanced_query = f"medical research {query}" if self._is_medical_query(query) else query
            
            response = self.client.search(
                query=enhanced_query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=["nih.gov", "who.int", "thelancet.com", "nejm.org", "jamanetwork.com"]
            )
            return response
        except Exception as e:
            logging.error(f"Error performing web search: {e}")
            return {"results": []}
    
    def format_search_results(self, search_results: Dict[str, Any]) -> str:
        """Format search results for context with medical research focus"""
        if not search_results.get("results"):
            return "No relevant medical research found in web search."
        
        formatted_results = "Recent medical research from web search:\n\n"
        for i, result in enumerate(search_results["results"], 1):
            formatted_results += f"Research {i}:\n"
            formatted_results += f"Title: {result.get('title', 'N/A')}\n"
            formatted_results += f"Content: {result.get('content', 'N/A')}\n"
            formatted_results += f"URL: {result.get('url', 'N/A')}\n\n"
        
        return formatted_results
    
    def get_search_results_with_urls(self, search_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Return search results with titles and URLs for display"""
        if not search_results.get("results"):
            return []
        
        return [
            {
                "title": result.get("title", "Untitled"),
                "url": result.get("url", "#")
            }
            for result in search_results["results"]
        ]
    
    def _is_medical_query(self, query: str) -> bool:
        """Determine if the query is medical in nature"""
        medical_terms = [
            'disease', 'treatment', 'patient', 'medical', 'health', 
            'medicine', 'clinical', 'symptom', 'diagnosis', 'therapy',
            'drug', 'pharmaceutical', 'hospital', 'doctor', 'nurse'
        ]
        
        query_lower = query.lower()
        return any(term in query_lower for term in medical_terms)