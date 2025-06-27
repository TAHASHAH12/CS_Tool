import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from urllib.parse import urljoin, urlparse
import re
from typing import List, Dict, Tuple
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit page
st.set_page_config(
    page_title="Semantic Similarity Backlink Analyzer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'cache' not in st.session_state:
    st.session_state.cache = {}

@st.cache_resource
def load_model():
    """Load the sentence transformer model with caching"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_content(url: str) -> str:
    """Extract text content from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text from main content areas
        content_selectors = ['main', 'article', '.content', '#content', '.post', '.entry']
        content = ""
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text() for elem in elements])
                break
        
        if not content:
            content = soup.get_text()
        
        # Clean and normalize text
        content = re.sub(r'\s+', ' ', content).strip()
        return content[:5000]  # Limit content length
        
    except Exception as e:
        st.error(f"Error extracting content from {url}: {str(e)}")
        return ""

def calculate_similarity(text1: str, text2: str, model) -> float:
    """Calculate cosine similarity between two texts"""
    if not text1 or not text2:
        return 0.0
    
    try:
        embeddings = model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return 0.0

def get_domain_pages(domain_url: str, max_pages: int = 10) -> List[str]:
    """Extract up to max_pages URLs from a domain"""
    try:
        parsed_url = urlparse(domain_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(domain_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a', href=True)
        
        urls = set()
        urls.add(domain_url)  # Include the original URL
        
        for link in links:
            href = link['href']
            if href.startswith('/'):
                full_url = urljoin(base_domain, href)
            elif href.startswith('http') and parsed_url.netloc in href:
                full_url = href
            else:
                continue
            
            # Filter out common non-content URLs
            if any(x in full_url.lower() for x in ['#', 'javascript:', 'mailto:', '.pdf', '.jpg', '.png', '.gif']):
                continue
            
            urls.add(full_url)
            if len(urls) >= max_pages:
                break
        
        return list(urls)[:max_pages]
    
    except Exception as e:
        st.error(f"Error crawling domain {domain_url}: {str(e)}")
        return [domain_url]

def get_similarity_color(score: float) -> str:
    """Return color based on similarity score"""
    if score >= 0.6:
        return "🟢"
    elif score >= 0.4:
        return "🟡"
    elif score >= 0.3:
        return "🟠"
    else:
        return "🔴"

def get_similarity_label(score: float) -> str:
    """Return label based on similarity score"""
    if score >= 0.6:
        return "Excellent"
    elif score >= 0.4:
        return "Good"
    elif score >= 0.3:
        return "Acceptable"
    else:
        return "Poor"

# Main UI
st.title("🔗 Semantic Similarity Backlink Analyzer")
st.markdown("Evaluate backlink opportunities based on semantic similarity and topical relevance.")

# Sidebar for tool selection
st.sidebar.title("🛠️ Tool Selection")
tool_choice = st.sidebar.selectbox(
    "Choose Analysis Tool:",
    ["Single URL Comparison", "Bulk URL Analysis", "Full Domain Analysis"]
)

# Load model
with st.spinner("Loading AI model..."):
    if st.session_state.model is None:
        st.session_state.model = load_model()
    model = st.session_state.model

# Display similarity thresholds
st.sidebar.markdown("### 📊 Similarity Thresholds")
st.sidebar.markdown("""
- 🟢 **0.6+**: Excellent semantic relevance
- 🟡 **0.4-0.59**: Good semantic relevance  
- 🟠 **0.3-0.39**: Acceptable semantic relevance
- 🔴 **<0.3**: Poor semantic relevance
""")

# Tool 1: Single URL Comparison
if tool_choice == "Single URL Comparison":
    st.header("🎯 Single URL Comparison")
    st.markdown("Compare the semantic similarity between two specific URLs.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_url = st.text_input("Target URL (Your Content)", placeholder="https://example.com/your-page")
    
    with col2:
        external_url = st.text_input("External URL (Potential Backlink)", placeholder="https://external-site.com/their-page")
    
    if st.button("🔍 Analyze Similarity", type="primary"):
        if target_url and external_url:
            with st.spinner("Extracting and analyzing content..."):
                # Extract content
                target_content = extract_content(target_url)
                external_content = extract_content(external_url)
                
                if target_content and external_content:
                    # Calculate similarity
                    similarity = calculate_similarity(target_content, external_content, model)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Similarity Score", f"{similarity:.3f}")
                    
                    with col2:
                        st.metric("Quality Rating", get_similarity_label(similarity))
                    
                    with col3:
                        st.markdown(f"### {get_similarity_color(similarity)} Status")
                    
                    # Detailed analysis
                    st.subheader("📋 Detailed Analysis")
                    
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        st.markdown("**Target Content Preview:**")
                        st.text_area("", target_content[:500] + "...", height=150, disabled=True, key="target_content_preview")
                    
                    with analysis_col2:
                        st.markdown("**External Content Preview:**")
                        st.text_area("", external_content[:500] + "...", height=150, disabled=True, key="external_content_preview")
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    if similarity >= 0.6:
                        st.success("🎯 **Excellent Match!** This is a high-priority backlink opportunity. The semantic relevance is very strong.")
                    elif similarity >= 0.4:
                        st.info("👍 **Good Match!** This backlink would be valuable. Consider reaching out with content-focused messaging.")
                    elif similarity >= 0.3:
                        st.warning("⚠️ **Acceptable Match.** This could work but isn't ideal. Look for better opportunities first.")
                    else:
                        st.error("❌ **Poor Match.** This backlink would provide minimal SEO value due to low topical relevance.")
                
                else:
                    st.error("Could not extract content from one or both URLs. Please check the URLs and try again.")
        else:
            st.warning("Please enter both URLs to perform the analysis.")

# Tool 2: Bulk URL Analysis
elif tool_choice == "Bulk URL Analysis":
    st.header("📊 Bulk URL Analysis")
    st.markdown("Analyze multiple URL pairs at once using CSV upload or manual entry.")
    
    # Option to choose input method
    input_method = st.radio("Choose input method:", ["CSV Upload", "Manual Entry"])
    
    if input_method == "CSV Upload":
        st.markdown("Upload a CSV file with columns: `target_url`, `external_url`")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            
            if st.button("🚀 Process All URLs", type="primary"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, row in df.iterrows():
                    status_text.text(f"Processing {i+1}/{len(df)}: {row['external_url']}")
                    
                    target_content = extract_content(row['target_url'])
                    external_content = extract_content(row['external_url'])
                    
                    if target_content and external_content:
                        similarity = calculate_similarity(target_content, external_content, model)
                    else:
                        similarity = 0.0
                    
                    results.append({
                        'Target URL': row['target_url'],
                        'External URL': row['external_url'],
                        'Similarity Score': similarity,
                        'Quality': get_similarity_label(similarity),
                        'Status': get_similarity_color(similarity)
                    })
                    
                    progress_bar.progress((i + 1) / len(df))
                
                status_text.text("Analysis complete!")
                
                # Display results
                results_df = pd.DataFrame(results)
                st.subheader("📈 Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Score", f"{results_df['Similarity Score'].mean():.3f}")
                with col2:
                    st.metric("Excellent (0.6+)", len(results_df[results_df['Similarity Score'] >= 0.6]))
                with col3:
                    st.metric("Good (0.4+)", len(results_df[results_df['Similarity Score'] >= 0.4]))
                with col4:
                    st.metric("Poor (<0.3)", len(results_df[results_df['Similarity Score'] < 0.3]))
                
                # Visualization
                fig = px.histogram(results_df, x='Similarity Score', nbins=20, 
                                 title="Distribution of Similarity Scores")
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button("📥 Download Results", csv, "similarity_results.csv", "text/csv")
    
    else:  # Manual Entry
        st.markdown("Enter URL pairs manually:")
        
        # Initialize session state for manual entries
        if 'manual_entries' not in st.session_state:
            st.session_state.manual_entries = []
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            target_url = st.text_input("Target URL", key="manual_target_url")
        with col2:
            external_url = st.text_input("External URL", key="manual_external_url")
        with col3:
            if st.button("➕ Add Pair"):
                if target_url and external_url:
                    st.session_state.manual_entries.append((target_url, external_url))
                    st.success("Added!")
        
        if st.session_state.manual_entries:
            st.subheader("📝 URL Pairs to Analyze")
            for i, (target, external) in enumerate(st.session_state.manual_entries):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.text(target)
                with col2:
                    st.text(external)
                with col3:
                    if st.button("🗑️", key=f"remove_{i}"):
                        st.session_state.manual_entries.pop(i)
                        st.rerun()
            
            if st.button("🚀 Analyze All Pairs", type="primary"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (target_url, external_url) in enumerate(st.session_state.manual_entries):
                    status_text.text(f"Processing {i+1}/{len(st.session_state.manual_entries)}")
                    
                    target_content = extract_content(target_url)
                    external_content = extract_content(external_url)
                    
                    if target_content and external_content:
                        similarity = calculate_similarity(target_content, external_content, model)
                    else:
                        similarity = 0.0
                    
                    results.append({
                        'Target URL': target_url,
                        'External URL': external_url,
                        'Similarity Score': similarity,
                        'Quality': get_similarity_label(similarity),
                        'Status': get_similarity_color(similarity)
                    })
                    
                    progress_bar.progress((i + 1) / len(st.session_state.manual_entries))
                
                status_text.text("Analysis complete!")
                
                # Display results (same as CSV upload)
                results_df = pd.DataFrame(results)
                st.subheader("📈 Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Score", f"{results_df['Similarity Score'].mean():.3f}")
                with col2:
                    st.metric("Excellent (0.6+)", len(results_df[results_df['Similarity Score'] >= 0.6]))
                with col3:
                    st.metric("Good (0.4+)", len(results_df[results_df['Similarity Score'] >= 0.4]))
                with col4:
                    st.metric("Poor (<0.3)", len(results_df[results_df['Similarity Score'] < 0.3]))
                
                # Visualization
                fig = px.histogram(results_df, x='Similarity Score', nbins=20, 
                                 title="Distribution of Similarity Scores")
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button("📥 Download Results", csv, "similarity_results.csv", "text/csv")

# Tool 3: Full Domain Analysis
elif tool_choice == "Full Domain Analysis":
    st.header("🌐 Full Domain Analysis (URLMatcher)")
    st.markdown("Discover the best page matches by crawling up to 10 pages from each domain.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_domain = st.text_input("Target Domain/URL", placeholder="https://yourdomain.com")
        target_pages = st.slider("Max pages to crawl (Target)", 1, 10, 5)
    
    with col2:
        external_domain = st.text_input("External Domain/URL", placeholder="https://external-site.com")
        external_pages = st.slider("Max pages to crawl (External)", 1, 10, 5)
    
    if st.button("🔍 Analyze Domains", type="primary"):
        if target_domain and external_domain:
            with st.spinner("Crawling domains and analyzing content..."):
                # Get pages from both domains
                target_urls = get_domain_pages(target_domain, target_pages)
                external_urls = get_domain_pages(external_domain, external_pages)
                
                st.info(f"Found {len(target_urls)} target pages and {len(external_urls)} external pages")
                
                # Extract content for all pages
                target_contents = {}
                external_contents = {}
                
                progress_bar = st.progress(0)
                total_pages = len(target_urls) + len(external_urls)
                current_page = 0
                
                for url in target_urls:
                    target_contents[url] = extract_content(url)
                    current_page += 1
                    progress_bar.progress(current_page / total_pages)
                
                for url in external_urls:
                    external_contents[url] = extract_content(url)
                    current_page += 1
                    progress_bar.progress(current_page / total_pages)
                
                # Calculate all similarities
                similarities = []
                for target_url, target_content in target_contents.items():
                    if not target_content:
                        continue
                    for external_url, external_content in external_contents.items():
                        if not external_content:
                            continue
                        
                        similarity = calculate_similarity(target_content, external_content, model)
                        similarities.append({
                            'Target URL': target_url,
                            'External URL': external_url,
                            'Similarity Score': similarity,
                            'Quality': get_similarity_label(similarity),
                            'Status': get_similarity_color(similarity)
                        })
                
                # Sort by similarity score
                similarities.sort(key=lambda x: x['Similarity Score'], reverse=True)
                
                # Display results
                st.subheader("🏆 Best Matches (Top 10)")
                top_matches = similarities[:10]
                
                for i, match in enumerate(top_matches, 1):
                    with st.expander(f"{i}. {match['Status']} Score: {match['Similarity Score']:.3f} - {match['Quality']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Target:** {match['Target URL']}")
                        with col2:
                            st.markdown(f"**External:** {match['External URL']}")
                
                # Full results table
                st.subheader("📊 All Results")
                results_df = pd.DataFrame(similarities)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Combinations", len(similarities))
                with col2:
                    st.metric("Best Score", f"{max(similarities, key=lambda x: x['Similarity Score'])['Similarity Score']:.3f}")
                with col3:
                    st.metric("Average Score", f"{np.mean([s['Similarity Score'] for s in similarities]):.3f}")
                with col4:
                    excellent_count = len([s for s in similarities if s['Similarity Score'] >= 0.6])
                    st.metric("Excellent Matches", excellent_count)
                
                # Heatmap visualization
                if len(target_urls) <= 10 and len(external_urls) <= 10:
                    st.subheader("🔥 Similarity Heatmap")
                    
                    # Create similarity matrix
                    matrix = np.zeros((len(target_urls), len(external_urls)))
                    for i, target_url in enumerate(target_urls):
                        for j, external_url in enumerate(external_urls):
                            for sim in similarities:
                                if sim['Target URL'] == target_url and sim['External URL'] == external_url:
                                    matrix[i][j] = sim['Similarity Score']
                                    break
                    
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=matrix,
                        x=[url.split('/')[-1][:20] + '...' if len(url.split('/')[-1]) > 20 else url.split('/')[-1] for url in external_urls],
                        y=[url.split('/')[-1][:20] + '...' if len(url.split('/')[-1]) > 20 else url.split('/')[-1] for url in target_urls],
                        colorscale='RdYlGn',
                        colorbar=dict(title="Similarity Score")
                    ))
                    fig.update_layout(title="URL Similarity Matrix", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button("📥 Download Full Results", csv, "domain_analysis_results.csv", "text/csv")
        
        else:
            st.warning("Please enter both domain URLs to perform the analysis.")

# Footer
st.markdown("---")
st.markdown("### 📚 About This Tool")
st.markdown("""
This tool uses advanced semantic similarity analysis to evaluate backlink opportunities based on topical relevance rather than just keyword matching. 
It leverages state-of-the-art sentence transformers to understand the contextual meaning of web page content.

**Key Benefits:**
- Prioritize high-quality backlink opportunities
- Focus outreach efforts on semantically relevant sites  
- Improve overall backlink profile quality
- Better understand content relationships

**Developed by Taha Shah**
""")