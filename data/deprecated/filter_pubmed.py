from __future__ import annotations

from datasets import load_dataset, load_from_disk
import pandas as pd
import random
import numpy as np
import os 
import sys
import json
import argparse
import fasttext
import requests
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Iterable, Tuple, Union

from tqdm import tqdm

class PubMedProcessor:
    METADATA_URL = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    MAX_RETRIES = 3

    def __init__(self, fasttext_model_path: str):
        self.fasttext_model_path = fasttext_model_path
        self.fasttext_model = fasttext.load_model(fasttext_model_path)
        # Create session for connection pooling and keep-alive
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
    
    @staticmethod
    def split_front_body_refs(text: str):
        if '=== Body' not in text or '==== Refs' not in text:
            return None
        
        front, body, refs = text.split('=== Body', 1)[0], text.split('=== Body', 1)[1].split('==== Refs\n', 1)[0], text.split('==== Refs\n', 1)[1]
        front = front.replace("==== Front", "")

        return front, body, refs

    def classify_lang (self, body: str):
        pred = self.fasttext_model.predict(body.replace("\n", " "))
        lang, conf = pred[0][0][0].replace("__label__", ""), pred[1][0][0]

        return lang, conf
    
    @staticmethod
    def _text(el: Optional[ET.Element]) -> str:
        return (el.text or "").strip() if el is not None else ""
    
    @staticmethod
    def _find(root: ET.Element, path: str) -> Optional[ET.Element]:
        return root.find(path)
    
    @staticmethod
    def _findall(root: ET.Element, path: str) -> List[ET.Element]:
        return root.findall(path)
    
    def fetch_metadata(self, pmid: Union[str, int]):
        def parse_article_title(article: ET.Element) -> str:
            """Return the article title."""
            return self._text(self._find(article, "./MedlineCitation/Article/ArticleTitle"))

        def parse_journal_title(article: ET.Element) -> str:
            """Return the Journal/Title string."""
            return self._text(self._find(article, "./MedlineCitation/Article/Journal/Title"))

        def parse_journal_metadata(article: ET.Element) -> Dict[str, Any]:
            """
            Extract journal volume, issue, ISSN, and ISO abbreviation.
            Returns: {"volume": str, "issue": str, "issn": str, "issn_type": str, "iso_abbreviation": str}
            """
            journal_issue = self._find(article, "./MedlineCitation/Article/Journal/JournalIssue")
            volume = self._text(self._find(journal_issue, "./Volume")) if journal_issue is not None else ""
            issue = self._text(self._find(journal_issue, "./Issue")) if journal_issue is not None else ""

            issn_elem = self._find(article, "./MedlineCitation/Article/Journal/ISSN")
            issn = self._text(issn_elem) if issn_elem is not None else ""
            issn_type = issn_elem.get("IssnType", "") if issn_elem is not None else ""

            iso_abbrev = self._text(self._find(article, "./MedlineCitation/Article/Journal/ISOAbbreviation"))

            return {
                "volume": volume,
                "issue": issue,
                "issn": issn,
                "issn_type": issn_type,
                "iso_abbreviation": iso_abbrev
            }

        def parse_pagination(article: ET.Element) -> Dict[str, str]:
            """
            Extract page information.
            Returns: {"start_page": str, "end_page": str, "medline_pgn": str}
            """
            pagination = self._find(article, "./MedlineCitation/Article/Pagination")
            if pagination is None:
                return {"start_page": "", "end_page": "", "medline_pgn": ""}

            return {
                "start_page": self._text(self._find(pagination, "./StartPage")),
                "end_page": self._text(self._find(pagination, "./EndPage")),
                "medline_pgn": self._text(self._find(pagination, "./MedlinePgn"))
            }

        def parse_publication_date(article: ET.Element) -> Dict[str, str]:
            """
            Extract publication date from JournalIssue/PubDate.
            Returns: {"year": str, "month": str, "day": str}
            """
            pubdate = self._find(article, "./MedlineCitation/Article/Journal/JournalIssue/PubDate")
            if pubdate is None:
                return {"year": "", "month": "", "day": ""}

            return {
                "year": self._text(self._find(pubdate, "./Year")),
                "month": self._text(self._find(pubdate, "./Month")),
                "day": self._text(self._find(pubdate, "./Day"))
            }

        def parse_article_date(article: ET.Element) -> Dict[str, Any]:
            """
            Extract electronic publication date.
            Returns: {"year": str, "month": str, "day": str, "date_type": str}
            """
            article_date = self._find(article, "./MedlineCitation/Article/ArticleDate")
            if article_date is None:
                return {"year": "", "month": "", "day": "", "date_type": ""}

            return {
                "year": self._text(self._find(article_date, "./Year")),
                "month": self._text(self._find(article_date, "./Month")),
                "day": self._text(self._find(article_date, "./Day")),
                "date_type": article_date.get("DateType", "")
            }

        def parse_mesh_headings(article: ET.Element) -> List[Dict[str, Any]]:
            """
            Extract MeSH (Medical Subject Headings) terms.
            Returns list of dicts with descriptor and qualifiers.
            """
            mesh_list = []
            for mesh in self._findall(article, "./MedlineCitation/MeshHeadingList/MeshHeading"):
                descriptor_elem = self._find(mesh, "./DescriptorName")
                if descriptor_elem is not None:
                    descriptor = {
                        "name": self._text(descriptor_elem),
                        "ui": descriptor_elem.get("UI", ""),
                        "major_topic": descriptor_elem.get("MajorTopicYN", "N")
                    }

                    # Get qualifiers
                    qualifiers = []
                    for qual in self._findall(mesh, "./QualifierName"):
                        qualifiers.append({
                            "name": self._text(qual),
                            "ui": qual.get("UI", ""),
                            "major_topic": qual.get("MajorTopicYN", "N")
                        })

                    mesh_list.append({
                        "descriptor": descriptor,
                        "qualifiers": qualifiers
                    })

            return mesh_list

        def parse_keywords(article: ET.Element) -> List[Dict[str, str]]:
            """
            Extract author-provided keywords.
            Returns list of dicts with keyword and major topic indicator.
            """
            keywords = []
            for kw in self._findall(article, "./MedlineCitation/KeywordList/Keyword"):
                keywords.append({
                    "keyword": self._text(kw),
                    "major_topic": kw.get("MajorTopicYN", "N")
                })
            return keywords

        def parse_publication_types(article: ET.Element) -> List[Dict[str, str]]:
            """
            Extract publication type information.
            Returns list of dicts with type name and UI.
            """
            pub_types = []
            for pt in self._findall(article, "./MedlineCitation/Article/PublicationTypeList/PublicationType"):
                pub_types.append({
                    "type": self._text(pt),
                    "ui": pt.get("UI", "")
                })
            return pub_types

        def parse_language(article: ET.Element) -> str:
            """Extract article language code."""
            return self._text(self._find(article, "./MedlineCitation/Article/Language"))

        def parse_chemicals(article: ET.Element) -> List[Dict[str, str]]:
            """
            Extract chemical/substance information.
            Returns list of dicts with substance name, UI, and registry number.
            """
            chemicals = []
            for chem in self._findall(article, "./MedlineCitation/ChemicalList/Chemical"):
                substance = self._find(chem, "./NameOfSubstance")
                chemicals.append({
                    "name": self._text(substance) if substance is not None else "",
                    "ui": substance.get("UI", "") if substance is not None else "",
                    "registry_number": self._text(self._find(chem, "./RegistryNumber"))
                })
            return chemicals

        def parse_references(article: ET.Element) -> List[Dict[str, Any]]:
            """
            Return a list of references.
            Each reference: {"citation": str, "ids": [{"type": "...","value": "..."}]}
            """
            refs: List[Dict[str, Any]] = []
            for ref in self._findall(article, "./PubmedData/ReferenceList/Reference"):
                citation = self._text(self._find(ref, "./Citation"))
                ids = []
                for rid in self._findall(ref, "./ArticleIdList/ArticleId"):
                    ids.append({"type": rid.get("IdType") or "", "value": self._text(rid)})
                refs.append({"citation": citation, "ids": ids})

            return refs

        def parse_article_ids(article: ET.Element) -> Dict[str, str]:
            """
            Map ArticleIdList -> {id_type: value}, e.g., {"pubmed": "...", "pmc": "...", "doi": "...", "pii": "..."}
            """
            out: Dict[str, str] = {}
            for aid in self._findall(article, "./PubmedData/ArticleIdList/ArticleId"):
                k = (aid.get("IdType") or "").strip().lower()
                v = self._text(aid)
                if k and v:
                    out[k] = v
            # Ensure PMID exists even if only in MedlineCitation
            if "pubmed" not in out:
                pmid_val = self._text(self._find(article, "./MedlineCitation/PMID"))
                if pmid_val:
                    out["pubmed"] = int(pmid_val)
            return out

        def parse_authors(article: ET.Element) -> List[Dict[str, Any]]:
            """
            Return a list of authors with name parts, affiliations, and identifiers.
            Each author: {"last":..., "fore":..., "initials":..., "affiliations":[...], "identifiers": [...]}
            """
            out: List[Dict[str, Any]] = []
            for au in self._findall(article, "./MedlineCitation/Article/AuthorList/Author"):
                last = self._text(self._find(au, "./LastName"))
                fore = self._text(self._find(au, "./ForeName"))
                initials = self._text(self._find(au, "./Initials"))
                collective = self._text(self._find(au, "./CollectiveName"))

                affs = [self._text(self._find(a, "./Affiliation")) for a in self._findall(au, "./AffiliationInfo")]
                affs = [a for a in affs if a]

                # Parse author identifiers (e.g., ORCID)
                identifiers = []
                for ident in self._findall(au, "./Identifier"):
                    identifiers.append({
                        "value": self._text(ident),
                        "source": ident.get("Source", "")
                    })

                out.append({
                    "last": last,
                    "fore": fore,
                    "initials": initials,
                    "collective_name": collective,
                    "affiliations": affs,
                    "identifiers": identifiers
                })
            return out

        def parse_abstract(article: ET.Element) -> Dict[str, Any]:
            """
            Return the full abstract text with labels preserved.
            Returns: {"text": str, "sections": [{"label": str, "text": str}], "copyright": str}
            """
            sections = []
            full_text_parts = []

            for ab in self._findall(article, "./MedlineCitation/Article/Abstract/AbstractText"):
                label = ab.get("Label", "")
                text = "".join(ab.itertext()).strip()
                if text:
                    sections.append({"label": label, "text": text})
                    full_text_parts.append(text)

            # Get copyright info
            copyright_info = self._text(self._find(article, "./MedlineCitation/Article/Abstract/CopyrightInformation"))

            # If no structured sections, try getting plain abstract
            if not full_text_parts:
                raw = (article.findtext("./MedlineCitation/Article/Abstract") or "").strip()
                return {"text": raw, "sections": [], "copyright": copyright_info}

            return {
                "text": "\n\n".join(full_text_parts),
                "sections": sections,
                "copyright": copyright_info
            }

        def parse_pubmed_xml(xml_text: str) -> List[Dict[str, Any]]:
            """
            Parse one or more <PubmedArticle> from a PubMed/MEDLINE XML string.
            Returns a list of dicts, one per article, with requested fields merged.
            """
            root = ET.fromstring(xml_text)
            records = []
            for art in root.findall("./PubmedArticle"):
                # Parse journal metadata
                journal_meta = parse_journal_metadata(art)
                pub_date = parse_publication_date(art)
                article_date = parse_article_date(art)
                pagination = parse_pagination(art)
                abstract_data = parse_abstract(art)

                rec = {
                    # Core identifiers
                    "pmid": int(pmid),
                    "article_ids": parse_article_ids(art),

                    # Article metadata
                    "article_title": parse_article_title(art),
                    "abstract": abstract_data.get("text", ""),
                    "abstract_sections": abstract_data.get("sections", []),
                    "abstract_copyright": abstract_data.get("copyright", ""),
                    "language": parse_language(art),

                    # Journal information
                    "journal_title": parse_journal_title(art),
                    "journal_iso_abbreviation": journal_meta["iso_abbreviation"],
                    "journal_issn": journal_meta["issn"],
                    "journal_issn_type": journal_meta["issn_type"],
                    "journal_volume": journal_meta["volume"],
                    "journal_issue": journal_meta["issue"],

                    # Pagination
                    "page_start": pagination["start_page"],
                    "page_end": pagination["end_page"],
                    "page_medline_pgn": pagination["medline_pgn"],

                    # Publication dates
                    "pub_date_year": pub_date["year"],
                    "pub_date_month": pub_date["month"],
                    "pub_date_day": pub_date["day"],
                    "article_date_year": article_date["year"],
                    "article_date_month": article_date["month"],
                    "article_date_day": article_date["day"],
                    "article_date_type": article_date["date_type"],

                    # Authors and affiliations
                    "authors": parse_authors(art),

                    # Classification and keywords
                    "mesh_headings": parse_mesh_headings(art),
                    "keywords": parse_keywords(art),
                    "publication_types": parse_publication_types(art),

                    # Chemical/substance information
                    "chemicals": parse_chemicals(art),

                    # References and related articles
                    "references": parse_references(art)
                }
                records.append(rec)
            return records

        if int(pmid) == 0:
            return []
        
        for _ in range(self.MAX_RETRIES):
            params = {
                    "db": "pubmed",
                    "id": str(pmid),
                    "rettype": "xml",
                    "retmode": "text"
                }
            r = self.session.get(self.METADATA_URL, params=params, timeout=15)

            if r.status_code == 200:
                break
            else:
                time.sleep(0.1)
        
        records = parse_pubmed_xml(r.text)

        return records

def testing(output_dir: str, num_samples:int = 1000):
    results = []
    pubmed = load_dataset("pmc/open_access")
    sample_ids = random.sample(range(1, len(pubmed["train"])), num_samples)
    
    processor = PubMedProcessor(fasttext_model_path="/mnt/home/al2644/storage/fasttext/models/lid.176.bin")

    for sample_id in tqdm(sample_ids):
        sample = pubmed["train"][sample_id]
        pmid = int(sample["pmid"])
        
        try:
            records = processor.fetch_metadata(pmid)

            if not records:
                error_record = {
                    "sample_id": sample_id,
                    "pmid": pmid,
                    "error": "No records returned from PubMed"
                }
                results.append(error_record)
            else:
                for record in records:
                    record["sample_id"] = sample_id
                results.extend(records)

        except Exception as e:
            error_record = {
                "sample_id": sample_id,
                "pmid": pmid,
                "error": str(e)
            }
            results.append(error_record)

    results = pd.DataFrame(results)
    results.to_parquet(os.path.join(output_dir, "pubmed_metadata.parquet"))

if __name__ == "__main__":
    output_dir = "data/pubmed_metadata"
    os.makedirs(output_dir, exist_ok=True)
    testing(output_dir, num_samples=500)