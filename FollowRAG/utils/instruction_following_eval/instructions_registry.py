from utils.instruction_following_eval import instructions


INSTRUCTION_DICT = {
    'keywords_inclusion': instructions.KeywordChecker,
    'keywords_frequency': instructions.KeywordFrequencyChecker,
    'keywords_exclusion': instructions.ForbiddenWords,
    'format_language': instructions.ResponseLanguageChecker,
    'length_sentence': instructions.NumberOfSentences,
    'length_paragraph': instructions.ParagraphChecker,
    'length_words': instructions.NumberOfWords,
    'position_first_word': instructions.ParagraphFirstWordCheck,
    'structure_placeholder': instructions.PlaceholderChecker,
    'position_postscript': instructions.PostscriptChecker,
    'structure_bullets': instructions.BulletListChecker,
    'structure_highlights': instructions.HighlightSectionChecker,
    'structure_sections': instructions.SectionChecker,
    'format_json': instructions.JsonFormat,
    'structure_title': instructions.TitleChecker,
    'format_repeat_question': instructions.RepeatPromptThenAnswer,
    'position_end_with': instructions.EndChecker,
    'cases_capital_words': instructions.CapitalWordFrequencyChecker,
    'cases_uppercase': instructions.CapitalLettersEnglishChecker,
    'cases_lowercase': instructions.LowercaseLettersEnglishChecker,
    'format_no_commas': instructions.CommaChecker,
    'format_quotation': instructions.QuotationChecker,
}

