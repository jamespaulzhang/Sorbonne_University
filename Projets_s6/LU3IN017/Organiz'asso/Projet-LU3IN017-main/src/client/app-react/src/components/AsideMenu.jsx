function AsideMenu (props) {

    return <>
        <div className="box">
            <h3>Forums</h3>
            <ul>
                <li>{props.currentPage.page !== "feed_page" ?
                    <a onClick={props.toFeedPage}>Forum membres</a> : "Forum membres"}</li>
                <li>{props.currentPage.page !== "feed_admin_page" ?
                    <a onClick={props.toFeedAdminPage}>Forum administrateurs</a> : "Forum administrateurs"}</li>
            </ul>
        </div>
    </>;

}

export default AsideMenu;