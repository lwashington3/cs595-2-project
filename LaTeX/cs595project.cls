%! Class = cs595project
%! Author = Len Washington III
%! Date = 3/16/26

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cs595project}

\RequirePackage{kvoptions}

\SetupKeyvalOptions{
	family=csproject,
	prefix=csproject@
}

\DeclareStringOption{title}

\ProcessKeyvalOptions*

\LoadClass[12pt]{report}

% Packages
\PassOptionsToPackage{sorting=none}{biblatex}
\PassOptionsToPackage{style=numeric}{iitacademics}
\usepackage[title={\csproject@title}]{cs595}

\addbibresource{project.bib}
\renewcommand{\thesection}{\arabic{section}.}
\renewcommand{\cite}{\parencite}

\newboolean{details}
\setboolean{details}{false}

\RequirePackage{multirow}
\RequirePackage{minted}

\RequirePackage{fontawesome5}
\definecolor{orcid}{HTML}{a6ce39}
\newcommand{\orcid}[1]{\href{https://orcid.org/#1}{\textcolor{orcid}{\faIcon{orcid}}}}

\newcolumntype{L}[1]{>{\raggedright\arraybackslash}m{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}m{#1}}

\let\oldtabular\tabular
\let\oldendtabular\endtabular

\renewenvironment{tabular}[1]
{
	\begin{lrbox}{\@tempboxa}
	\oldtabular{#1}
}{
	\oldendtabular
	\end{lrbox}
	\centerline{\usebox{\@tempboxa}}
}