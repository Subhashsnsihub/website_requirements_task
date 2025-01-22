import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse
from openai import OpenAI
import time
import logging
import pytesseract
from PIL import Image
from io import BytesIO
import re
import cv2
import numpy as np
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Tesseract path for Windows
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class WebScraperQA:
    def __init__(self, api_key: str):
        """Initialize the WebScraperQA system with OpenAI API key."""
        self.visited_urls = set()
        self.data = {}
        self.client = OpenAI(api_key=api_key)
        self.extracted_metrics = {}
        
    def is_valid_url(self, url: str, domain: str) -> bool:
        """Check if URL is valid and belongs to the same domain."""
        try:
            parsed = urlparse(url)
            return parsed.netloc == domain and parsed.scheme in ['http', 'https']
        except:
            return False
            
    def extract_image_text(self, img_url: str) -> str:
        """Extract text from images using OCR."""
        try:
            response = requests.get(img_url, timeout=10)
            image = Image.open(BytesIO(response.content))
            
            opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            text = pytesseract.image_to_string(thresh)
            self.extract_metrics_from_text(text)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from image {img_url}: {str(e)}")
            return ""

    def extract_contact_info(self, text: str) -> dict:
        """Extract contact information using regex patterns."""
        contact_info = {
            'emails': [],
            'phones': [],
            'addresses': [],
            'social_media': [],
            'working_hours': []
        }
        
        # Email pattern
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text)
        contact_info['emails'] = list(set(emails))
        
        # Phone pattern
        phone_pattern = r'(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4})'
        phones = re.findall(phone_pattern, text)
        contact_info['phones'] = list(set(phones))
        
        # Social media handles
        social_patterns = {
            'facebook': r'(?:facebook\.com|fb\.com)/[\w.]+',
            'twitter': r'(?:twitter\.com|x\.com)/[\w]+',
            'linkedin': r'linkedin\.com/(?:in|company)/[\w-]+',
            'instagram': r'instagram\.com/[\w.]+',
            'youtube': r'youtube\.com/[@\w]+'
        }
        
        for platform, pattern in social_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                contact_info['social_media'].extend([f"{platform}: {match}" for match in matches])
        
        # Address patterns
        address_indicators = [
            r'(?:Address|Location|Office)[\s:]+([^.]*(?:Road|Street|Avenue|Lane|Building|Campus|Highway|Plot|District|State|Country|Pincode|\d{6})[^.]*\.)',
            r'\d+[A-Za-z\s,]+(?:Road|Street|Avenue|Lane|Building|Campus|Highway)[A-Za-z\s,]+\d{6}',
        ]
        
        for pattern in address_indicators:
            addresses = re.findall(pattern, text, re.IGNORECASE)
            contact_info['addresses'].extend([addr.strip() for addr in addresses if len(addr.strip()) > 10])
        
        # Working hours pattern
        hours_patterns = [
            r'(?:Working Hours|Business Hours|Open|Hours)[\s:]+([^.]*(?:AM|PM|am|pm)[^.]*\.)',
            r'(?:Monday|Mon)[\s-]+(?:Friday|Fri)[^.]*(?:AM|PM|am|pm)[^.]*\.',
            r'\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)\s*[-–]\s*\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)'
        ]
        
        for pattern in hours_patterns:
            hours = re.findall(pattern, text, re.IGNORECASE)
            contact_info['working_hours'].extend([h.strip() for h in hours if len(h.strip()) > 5])
        
        # Remove duplicates while preserving order
        for key in contact_info:
            contact_info[key] = list(dict.fromkeys(contact_info[key]))
        
        return contact_info

    def extract_metrics_from_text(self, text: str) -> None:
        """Extract numerical metrics from text using regex patterns."""
        patterns = [
            (r'(\d+)\+?\s*(?:years?|yrs?)', 'years'),
            (r'(\d+)\+?\s*(?:students?|learners?)', 'students'),
            (r'(\d+)\+?\s*(?:faculty|professors?|teachers?)', 'faculty'),
            (r'(\d+)\+?\s*(?:labs?|laboratories?)', 'labs'),
            (r'(\d+)\+?\s*(?:programs?|courses?)', 'programs'),
            (r'(\d+)\+?\s*(?:scholarships?|awards?)', 'scholarships'),
            (r'(\d+)\+?\s*(?:alumni|graduates?)', 'alumni'),
            (r'(\d+)\+?\s*(?:MoUs?|partnerships?)', 'partnerships'),
            (r'(\d+)\+?\s*(?:centres?|centers?)', 'centers'),
            (r'(\d+)\+?\s*(?:patents?|innovations?)', 'patents'),
            (r'(\d+)\+?\s*(?:companies|recruiters?)', 'recruiters')
        ]
        
        for pattern, category in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                self.extracted_metrics[category] = max([int(x) for x in matches])

    def extract_page_data(self, url: str, soup: BeautifulSoup) -> dict:
        """Extract relevant data from a webpage."""
        data = {
            'url': url,
            'title': '',
            'metadata': {},
            'content': '',
            'headings': [],
            'links': [],
            'tables': [],
            'leadership': {},
            'image_text': [],
            'metrics': {},
            'contact_info': {}
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            data['title'] = title_tag.text.strip()
            
        # Extract metadata
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            name = tag.get('name', '').lower()
            content = tag.get('content', '')
            if name and content:
                data['metadata'][name] = content
                
        # Extract footer content
        footer_selectors = ['footer', '.footer', '#footer', '[class*="footer"]', '[id*="footer"]']
        footer_content = []
        
        for selector in footer_selectors:
            footer_elements = soup.select(selector)
            for element in footer_elements:
                footer_text = element.get_text(strip=True, separator=' ')
                footer_content.append(footer_text)
        
        # Extract contact information from footer
        combined_footer_text = ' '.join(footer_content)
        data['contact_info'] = self.extract_contact_info(combined_footer_text)
                
        # Process images and extract text
        images = soup.find_all('img')
        for img in images:
            img_url = img.get('src', '')
            if img_url:
                img_url = urljoin(url, img_url)
                img_text = self.extract_image_text(img_url)
                if img_text.strip():
                    data['image_text'].append({
                        'url': img_url,
                        'text': img_text
                    })
                    
        # Extract main content
        content_elements = []
        selectors = ['main', 'article', 'body', '.content', '#content', '.main-content']
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True, separator=' ')
                content_elements.append(text)
                self.extract_metrics_from_text(text)
                
        data['content'] = ' '.join(content_elements)
        
        # Extract leadership information
        leadership_keywords = {
            'chairman': ['chairman', 'chairperson'],
            'director': ['director', 'technical director'],
            'principal': ['principal'],
            'correspondent': ['correspondent'],
            'dean': ['dean']
        }
        
        for role, keywords in leadership_keywords.items():
            for keyword in keywords:
                for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    if keyword.lower() in heading.text.lower():
                        next_elem = heading.find_next()
                        while next_elem and not next_elem.text.strip():
                            next_elem = next_elem.find_next()
                        if next_elem:
                            data['leadership'][role] = next_elem.text.strip()
        
        # Add extracted metrics
        data['metrics'] = self.extracted_metrics.copy()
        
        return data
        
    def crawl_website(self, start_url: str, max_pages: int = 50) -> None:
        """Crawl website starting from given URL."""
        domain = urlparse(start_url).netloc
        urls_to_visit = [start_url]
        self.extracted_metrics = {}
        
        while urls_to_visit and len(self.visited_urls) < max_pages:
            url = urls_to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
                
            try:
                logger.info(f"Crawling: {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                self.data[url] = self.extract_page_data(url, soup)
                self.visited_urls.add(url)
                
                for link in soup.find_all('a', href=True):
                    new_url = urljoin(url, link['href'])
                    if (
                        new_url not in self.visited_urls and 
                        new_url not in urls_to_visit and
                        self.is_valid_url(new_url, domain)
                    ):
                        urls_to_visit.append(new_url)
                        
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error crawling {url}: {str(e)}")
                
    def answer_question(self, question: str) -> str:
        """Answer questions using GPT based on extracted website data."""
        question_lower = question.lower()
        
        # Check for contact information questions
        contact_keywords = ['contact', 'email', 'phone', 'address', 'location', 'social media', 
                          'facebook', 'twitter', 'linkedin', 'working hours', 'business hours']
        
        if any(keyword in question_lower for keyword in contact_keywords):
            for url, page_data in self.data.items():
                contact_info = page_data.get('contact_info', {})
                if contact_info:
                    relevant_info = []
                    if 'email' in question_lower and contact_info.get('emails'):
                        relevant_info.append(f"Email(s): {', '.join(contact_info['emails'])}")
                    if 'phone' in question_lower and contact_info.get('phones'):
                        relevant_info.append(f"Phone(s): {', '.join(contact_info['phones'])}")
                    if ('address' in question_lower or 'location' in question_lower) and contact_info.get('addresses'):
                        relevant_info.append(f"Address(es): {', '.join(contact_info['addresses'])}")
                    if 'social' in question_lower and contact_info.get('social_media'):
                        relevant_info.append(f"Social Media: {', '.join(contact_info['social_media'])}")
                    if ('hours' in question_lower or 'timing' in question_lower) and contact_info.get('working_hours'):
                        relevant_info.append(f"Working Hours: {', '.join(contact_info['working_hours'])}")
                    
                    if relevant_info:
                        return '\n'.join(relevant_info)
        
        # Check for metrics-related questions
        metric_patterns = {
            r'how many (\w+)': None,
            r'what is the (\w+)': None,
            r'number of (\w+)': None,
            r'total (\w+)': None
        }
        
        for pattern in metric_patterns:
            match = re.search(pattern, question_lower)
            if match:
                keyword = match.group(1)
                for url, page_data in self.data.items():
                    metrics = page_data.get('metrics', {})
                    for metric_name, value in metrics.items():
                        if keyword in metric_name.lower():
                            return f"Based on the extracted data, there are {value} {metric_name}."
        
        # Check for leadership questions
        leadership_keywords = ['chairman', 'correspondent', 'director', 'principal', 'dean']
        for keyword in leadership_keywords:
            if keyword in question_lower:
                for url, page_data in self.data.items():
                    if 'leadership' in page_data and page_data['leadership']:
                        for role, name in page_data['leadership'].items():
                            if keyword in role.lower():
                                return f"The {role} is {name}"
        
        # For other questions, combine all content and use GPT
        all_content = []
        for url, page_data in self.data.items():
            content = page_data.get('content', '')
            image_texts = [img['text'] for img in page_data.get('image_text', [])]
            footer_content = []
            for key, values in page_data.get('contact_info', {}).items():
                footer_content.extend([f"{key}: {', '.join(values)}"])
            
            combined_content = f"""
            URL: {url}
            Content: {content}
            Image Text: {' '.join(image_texts)}
            Footer Information: {' '.join(footer_content)}
            """
            all_content.append(combined_content)
            
        # Prepare context and question for GPT
        context = "\n".join(all_content)
        prompt = f"""Based on the following website content, please answer this question: {question}

Content:
{context}

Please provide a clear and concise answer based only on the information available in the content above."""

        try:
            # Call GPT for general questions
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided website content. Only use the information provided in the content to answer questions. If the information isn't available in the content, say so."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error getting GPT response: {str(e)}")
            return "I apologize, but I encountered an error while processing your question. Please try again."

def main():
    st.title("Website QA System")
    
    # Get OpenAI API key
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    
    # Get website URL
    website_url = st.text_input("Enter website URL to analyze:")
    
    if api_key and website_url:
        try:
            # Initialize WebScraperQA
            scraper = WebScraperQA(api_key)
            
            # Crawl website
            with st.spinner("Crawling website... This may take a few minutes."):
                scraper.crawl_website(website_url)
            
            st.success("Website crawling completed!")
            
            # Display extracted metrics
            if scraper.extracted_metrics:
                st.subheader("Extracted Metrics")
                for metric, value in scraper.extracted_metrics.items():
                    st.write(f"{metric.title()}: {value}")
            
            # Question answering interface
            st.subheader("Ask Questions")
            question = st.text_input("Enter your question about the website:")
            
            if question:
                with st.spinner("Finding answer..."):
                    answer = scraper.answer_question(question)
                    st.write("Answer:", answer)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()