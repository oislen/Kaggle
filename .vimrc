" Install Vundle
" git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
" useful links
" https://realpython.com/vim-and-python-a-match-made-in-heaven/
" https://www.freecodecamp.org/news/turning-vim-into-an-r-ide-cd9602e8c217/

set nocompatible
filetype off

set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

Plugin 'VundleVim/Vundle.vim'
Plugin 'itchyny/lightline.vim' "status bar
Plugin 'preservim/nerdtree' "folder tree
Plugin 'tpope/vim-fugitive' "git repo functionality
Plugin 'jewes/Conque-Shell' "interactive bash console
Plugin 'tmhedberg/SimpylFold' "enable code folding at indents
"Plugin 'jalvesaq/Nvim-R' "terminal interface for R
"Plugin 'jpalardy/vim-slime' "terminal interface for Python
"Plugin 'hanschen/vim-ipython-cell' "terminal interface for Python
Plugin 'kien/ctrlp.vim' "search functionality

call vundle#end()
filetype plugin indent on

"General Config settings
" set a colour column at 120
set colorcolumn=80
" enable line numbers
set number
":set relativenumber
" Enable folding
set foldmethod=indent
set foldlevel=99
" set python utf-8 encoding
set encoding=utf-8
" reset system clipboard
set clipboard=unnamed

" configure lightline settings
"stop lagging
set ttimeoutlen=30 
"show lightline bar
set laststatus=2 
" set colour scheme
if !has('gui_running')
	set t_Co=256
endif
"remove duplicate --INSERT-- mode
set noshowmode

" configure NERDTreeToggle
map <C-n> :NERDTreeToggle<CR>
" Enable folding with the spacebar
nnoremap <space> za

" configure conque-shell
" set default version of python
let g:ConqueTerm_PyVersion=3.7.3
" disable syntax colouring
let g:ConqueTerm_Color=0
" enable fast mode
let g:ConqueTerm_FastMode=1
